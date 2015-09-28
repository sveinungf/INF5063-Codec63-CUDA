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

static texture<uint8_t, 2, cudaReadModeElementType> tex_ref;

__global__
void block_sad(uint8_t* orig_block, int* block_sads, int stride, int xOffset, int yOffset)
{
	int i = threadIdx.x;
	int j = threadIdx.y;

	__shared__ uint8_t shared_orig_block[64];

	if (i < 8 && j < 8)
	{
		shared_orig_block[i*8 + j] = orig_block[i*stride + j];
	}

	__syncthreads();

	int result = 0;

	for (int y = 0; y < 8; ++y)
	{
		for (int x = 0; x < 8; ++x)
		{
			result += abs(tex2D(tex_ref, xOffset+i+x, yOffset+j+y) - shared_orig_block[y*8 + x]);
		}
	}

	block_sads[j*blockDim.x + i] = result;
}

static void sad_block_8x8_full_range(uint8_t* orig_block_gpu, int range_width, int range_height, int stride, int* pixel_sads_gpu, int* block_sads_gpu, int* block_sads, int xOffset, int yOffset)
{
	int numBlocks = 1;
	dim3 threadsPerBlock(range_width, range_height);
	block_sad<<<numBlocks, threadsPerBlock>>>(orig_block_gpu, block_sads_gpu, stride, xOffset, yOffset);

	const int block_sads_size = range_width * range_height * sizeof(int);
	cudaMemcpy(block_sads, block_sads_gpu, block_sads_size, cudaMemcpyDeviceToHost);
}

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *orig_gpu, int color_component, int* pixel_sads_gpu, int* block_sads_gpu, int* block_sads)
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
	sad_block_8x8_full_range(orig_block_gpu, range_width, range_height, w, pixel_sads_gpu, block_sads_gpu, block_sads, left, top);

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

	uint8_t *origY_gpu, *origU_gpu, *origV_gpu;
	int* pixel_sads_gpu;
	int* block_sads_gpu;
	int* block_sads;

	const int range = cm->me_search_range;
	const int max_range_width = 2 * range;
	const int max_range_height = 2 * range;
	const int pixel_sads_size = max_range_width * max_range_height * 64 * sizeof(int); // dat memory tho
	const int block_sads_size = max_range_width * max_range_height * sizeof(int);

	const int frame_size_Y = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t);
	const int frame_size_U = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t);
	const int frame_size_V = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t);

	cudaMalloc((void**) &pixel_sads_gpu, pixel_sads_size);
	cudaMalloc((void**) &block_sads_gpu, block_sads_size);
	block_sads = (int*) malloc(max_range_width * max_range_height * sizeof(int));

	cudaMalloc((void**) &origY_gpu, frame_size_Y);
	cudaMalloc((void**) &origU_gpu, frame_size_U);
	cudaMalloc((void**) &origV_gpu, frame_size_V);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();

	cudaArray* refY_array_gpu;
	cudaArray* refU_array_gpu;
	cudaArray* refV_array_gpu;

	cudaMallocArray(&refY_array_gpu, &channelDesc, cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT]);
	cudaMallocArray(&refU_array_gpu, &channelDesc, cm->padw[U_COMPONENT], cm->padh[U_COMPONENT]);
	cudaMallocArray(&refV_array_gpu, &channelDesc, cm->padw[V_COMPONENT], cm->padh[V_COMPONENT]);

	cudaMemcpy(origY_gpu, cm->curframe->orig->Y, frame_size_Y, cudaMemcpyHostToDevice);
	cudaMemcpy(origU_gpu, cm->curframe->orig->U, frame_size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(origV_gpu, cm->curframe->orig->V, frame_size_V, cudaMemcpyHostToDevice);

	cudaMemcpyToArray(refY_array_gpu, 0, 0, cm->refframe->recons->Y, frame_size_Y, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(refU_array_gpu, 0, 0, cm->refframe->recons->U, frame_size_U, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(refV_array_gpu, 0, 0, cm->refframe->recons->V, frame_size_V, cudaMemcpyHostToDevice);

	tex_ref.filterMode = cudaFilterModePoint;

	cudaBindTextureToArray(tex_ref, refY_array_gpu);

	/* Luma */
	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			me_block_8x8(cm, mb_x, mb_y, origY_gpu, Y_COMPONENT, pixel_sads_gpu, block_sads_gpu, block_sads);
		}
	}

	cudaUnbindTexture(tex_ref);
	cudaBindTextureToArray(tex_ref, refU_array_gpu);

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			me_block_8x8(cm, mb_x, mb_y, origU_gpu, U_COMPONENT, pixel_sads_gpu, block_sads_gpu, block_sads);
		}
	}

	cudaUnbindTexture(tex_ref);
	cudaBindTextureToArray(tex_ref, refV_array_gpu);

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			me_block_8x8(cm, mb_x, mb_y, origV_gpu, V_COMPONENT, pixel_sads_gpu, block_sads_gpu, block_sads);
		}
	}

	cudaUnbindTexture(tex_ref);

	cudaFree(pixel_sads_gpu);
	cudaFree(block_sads_gpu);
	free(block_sads);

	cudaFree(origY_gpu);
	cudaFree(origU_gpu);
	cudaFree(origV_gpu);

	cudaFreeArray(refY_array_gpu);
	cudaFreeArray(refU_array_gpu);
	cudaFreeArray(refV_array_gpu);
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
