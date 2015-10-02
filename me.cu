extern "C" {
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
}

#include "me.h"


__global__
void block_sad(uint8_t* orig_block, uint8_t* ref_search_range, int* block_sads, int stride)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int k = j*blockDim.x + i;

	__shared__ uint8_t shared_orig_block[64];

	if (i < 8 && j < 8)
	{
		shared_orig_block[j*8 + i] = orig_block[j*stride + i];
	}

	__syncthreads();

	uint8_t* ref_block = ref_search_range + j*stride + i;
	int result = 0;

	for (int y = 0; y < 8; ++y)
	{
		for (int x = 0; x < 8; ++x)
		{
			result += abs(ref_block[y*stride + x] - shared_orig_block[y*8 + x]);
		}
	}

	block_sads[k] = result;
}

static void sad_block_8x8_full_range(uint8_t* orig_block_gpu, uint8_t* ref_search_range_gpu, int range_width, int range_height, int stride, int* block_sads_gpu, int* block_sads)
{
	int numBlocks = 1;
	dim3 threadsPerBlock(range_width, range_height);
	block_sad<<<numBlocks, threadsPerBlock>>>(orig_block_gpu, ref_search_range_gpu, block_sads_gpu, stride);

	const int block_sads_size = range_width * range_height * sizeof(int);
	cudaMemcpy(block_sads, block_sads_gpu, block_sads_size, cudaMemcpyDeviceToHost);
}

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *orig_gpu, uint8_t *ref_gpu, int color_component, int* block_sads_gpu, int* block_sads)
{
	struct macroblock *mb = &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

	int range = cm->me_search_range;

	/* Quarter resolution for chroma channels. */
	if (color_component > 0)
	{
		range /= 2;
	}

	int left = mb_x * 8 - range;
	int top = mb_y * 8 - range;
	int right = mb_x * 8 + range;
	int bottom = mb_y * 8 + range;

	int w = cm->padw[color_component];
	int h = cm->padh[color_component];

	/* Make sure we are within bounds of reference frame. TODO: Support partial
	 frame bounds. */
	if (left < 0)
	{
		left = 0;
	}
	if (top < 0)
	{
		top = 0;
	}
	if (right > (w - 8))
	{
		right = w - 8;
	}
	if (bottom > (h - 8))
	{
		bottom = h - 8;
	}

	int mx = mb_x * 8;
	int my = mb_y * 8;

	int best_sad = INT_MAX;

	int range_width = right - left;
	int range_height = bottom - top;

	uint8_t* orig_block_gpu = orig_gpu + my * w + mx;
	uint8_t* ref_search_range_gpu = ref_gpu + top*w + left;
	sad_block_8x8_full_range(orig_block_gpu, ref_search_range_gpu, range_width, range_height, w, block_sads_gpu, block_sads);

	int p;
	for (p = 0; p < range_height; ++p)
	{
		int q;
		for (q = 0; q < range_width; ++q)
		{
			int sad = block_sads[p*range_width + q];

			if (sad < best_sad)
			{
				mb->mv_x = left + q - mx;
				mb->mv_y = top + p - my;
				best_sad = sad;
			}
		}
	}

	/* Here, there should be a threshold on SAD that checks if the motion vector
	 is cheaper than intraprediction. We always assume MV to be beneficial */

	mb->use_mv = 1;
}

void c63_motion_estimate(struct c63_common *cm)
{
	/* Compare this frame with previous reconstructed frame */
	int mb_x, mb_y;

	const int frame_size_Y = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t);
	const int frame_size_U = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t);
	const int frame_size_V = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t);

	cudaMemcpy(cm->cuda_me.origY_gpu, cm->curframe->orig->Y, frame_size_Y, cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.origU_gpu, cm->curframe->orig->U, frame_size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.origV_gpu, cm->curframe->orig->V, frame_size_V, cudaMemcpyHostToDevice);

	cudaMemcpy(cm->cuda_me.refY_gpu, cm->refframe->recons->Y, frame_size_Y, cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.refU_gpu, cm->refframe->recons->U, frame_size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(cm->cuda_me.refV_gpu, cm->refframe->recons->V, frame_size_V, cudaMemcpyHostToDevice);

	/* Luma */
	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			me_block_8x8(cm, mb_x, mb_y, cm->cuda_me.origY_gpu, cm->cuda_me.refY_gpu, Y_COMPONENT, cm->cuda_me.block_sads_gpu, cm->cuda_me.block_sads);
		}
	}

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			me_block_8x8(cm, mb_x, mb_y, cm->cuda_me.origU_gpu, cm->cuda_me.refU_gpu, U_COMPONENT, cm->cuda_me.block_sads_gpu, cm->cuda_me.block_sads);
			me_block_8x8(cm, mb_x, mb_y, cm->cuda_me.origV_gpu, cm->cuda_me.refV_gpu, V_COMPONENT, cm->cuda_me.block_sads_gpu, cm->cuda_me.block_sads);
		}
	}
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *predicted, uint8_t *ref, int color_component)
{
	struct macroblock *mb = &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

	if (!mb->use_mv)
	{
		return;
	}

	int left = mb_x * 8;
	int top = mb_y * 8;
	int right = left + 8;
	int bottom = top + 8;

	int w = cm->padw[color_component];

	/* Copy block from ref mandated by MV */
	int x, y;

	for (y = top; y < bottom; ++y)
	{
		for (x = left; x < right; ++x)
		{
			predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
		}
	}
}

void c63_motion_compensate(struct c63_common *cm)
{
	int mb_x, mb_y;

	/* Luma */
	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y, cm->refframe->recons->Y, Y_COMPONENT);
		}
	}

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U, cm->refframe->recons->U, U_COMPONENT);
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V, cm->refframe->recons->V, V_COMPONENT);
		}
	}
}
