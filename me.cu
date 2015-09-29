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
}

#include "dsp.h"
#include "me.h"


__global__
void block_sad(uint8_t* orig_block, uint8_t* ref_search_range, int* block_sads, int stride)
{
	int i = threadIdx.x;
	int j = threadIdx.y;

	__shared__ uint8_t shared_orig_block[64];

	if (i < 8 && j < 8)
	{
		shared_orig_block[i*8 + j] = orig_block[i*stride + j];
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

	block_sads[j*blockDim.x + i] = result;
}

__global__
void find_min(int* block_sads, int* result)
{
	int i = threadIdx.x * 2;

	__shared__ int indexes[1024];

	if (block_sads[i] > block_sads[i + 1])
	{
		block_sads[i] = block_sads[i + 1];
		indexes[i] = i + 1;
	}
	else
	{
		indexes[i] = i;
	}

	i *= 2;
	__syncthreads();

	if (i < 1024 && block_sads[i] > block_sads[i + 2])
	{
		block_sads[i] = block_sads[i + 2];
		indexes[i] = indexes[i + 2];
	}

	i *= 2;
	__syncthreads();

	if (i < 1024 && block_sads[i] > block_sads[i + 4]) // 1016 v 1020
	{
		block_sads[i] = block_sads[i + 4];
		indexes[i] = indexes[i + 4];
	}

	i *= 2;
	__syncthreads();

	if (i < 1024 && block_sads[i] > block_sads[i + 8]) // 1008 v 1016
	{
		block_sads[i] = block_sads[i + 8];
		indexes[i] = indexes[i + 8];
	}

	i *= 2;
	__syncthreads();

	if (i < 1024 && block_sads[i] > block_sads[i + 16]) // 992 v 1008
	{
		block_sads[i] = block_sads[i + 16];
		indexes[i] = indexes[i + 16];
	}

	i *= 2;
	__syncthreads();

	if (i < 1024 && block_sads[i] > block_sads[i + 32]) // 960 v 992
	{
		block_sads[i] = block_sads[i + 32];
		indexes[i] = indexes[i + 32];
	}

	i *= 2;
	__syncthreads();

	if (i < 1024 && block_sads[i] > block_sads[i + 64]) // 896 v 960, tid = 7
	{
		block_sads[i] = block_sads[i + 64];
		indexes[i] = indexes[i + 64];
	}

	i *= 2;
	__syncthreads();

	if (i < 1024 && block_sads[i] > block_sads[i + 128]) // tid = 3, 768 v 896
	{
		block_sads[i] = block_sads[i + 128];
		indexes[i] = indexes[i + 128];
	}

	i *= 2;
	__syncthreads();

	if (i < 1024 && block_sads[i] > block_sads[i + 256]) // tid = 0&1, 0v256 & 512v768
	{
		block_sads[i] = block_sads[i + 256];
		indexes[i] = indexes[i + 256];
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		*result = block_sads[0] > block_sads[512] ? indexes[512] : indexes[0];
	}
}

static void sad_block_8x8_full_range(uint8_t* orig_block_gpu, uint8_t* ref_search_range_gpu, int range_width, int range_height, int stride, int* pixel_sads_gpu, int* block_sads_gpu, int* result)
{
	int numBlocks = 1;
	dim3 threadsPerBlock(range_width, range_height);
	block_sad<<<numBlocks, threadsPerBlock>>>(orig_block_gpu, ref_search_range_gpu, block_sads_gpu, stride);

	int* result_gpu;
	cudaMalloc((void**) &result_gpu, sizeof(int));

	find_min<<<1, 512>>>(block_sads_gpu, result_gpu);
	cudaMemcpy(result, result_gpu, sizeof(int), cudaMemcpyDeviceToHost);
}

static void sad_block_8x8_semifull_range(uint8_t* orig_block_gpu, uint8_t* ref_search_range_gpu, int range_width, int range_height, int stride, int* pixel_sads_gpu, int* block_sads_gpu, int* block_sads)
{
	int numBlocks = 1;
	dim3 threadsPerBlock(range_width, range_height);
	block_sad<<<numBlocks, threadsPerBlock>>>(orig_block_gpu, ref_search_range_gpu, block_sads_gpu, stride);

	const int block_sads_size = range_width * range_height * sizeof(int);
	cudaMemcpy(block_sads, block_sads_gpu, block_sads_size, cudaMemcpyDeviceToHost);
}

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *orig_gpu, uint8_t *ref_gpu, int color_component, int* pixel_sads_gpu, int* block_sads_gpu, int* block_sads)
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

	if (range_width < 32 || range_height < 32)
	{
		sad_block_8x8_semifull_range(orig_block_gpu, ref_search_range_gpu, range_width, range_height, w, pixel_sads_gpu, block_sads_gpu, block_sads);

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
	}
	else
	{
		int result;
		sad_block_8x8_full_range(orig_block_gpu, ref_search_range_gpu, range_width, range_height, w, pixel_sads_gpu, block_sads_gpu, &result);

		mb->mv_x = left + (result%32) - mx;
		mb->mv_y = top + (result/32) - my;
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
	uint8_t *refY_gpu, *refU_gpu, *refV_gpu;
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

	cudaMalloc((void**) &refY_gpu, frame_size_Y);
	cudaMalloc((void**) &refU_gpu, frame_size_U);
	cudaMalloc((void**) &refV_gpu, frame_size_V);

	cudaMemcpy(origY_gpu, cm->curframe->orig->Y, frame_size_Y, cudaMemcpyHostToDevice);
	cudaMemcpy(origU_gpu, cm->curframe->orig->U, frame_size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(origV_gpu, cm->curframe->orig->V, frame_size_V, cudaMemcpyHostToDevice);

	cudaMemcpy(refY_gpu, cm->refframe->recons->Y, frame_size_Y, cudaMemcpyHostToDevice);
	cudaMemcpy(refU_gpu, cm->refframe->recons->U, frame_size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(refV_gpu, cm->refframe->recons->V, frame_size_V, cudaMemcpyHostToDevice);

	/* Luma */
	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			me_block_8x8(cm, mb_x, mb_y, origY_gpu, refY_gpu, Y_COMPONENT, pixel_sads_gpu, block_sads_gpu, block_sads);
		}
	}

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			me_block_8x8(cm, mb_x, mb_y, origU_gpu, refU_gpu, U_COMPONENT, pixel_sads_gpu, block_sads_gpu, block_sads);
			me_block_8x8(cm, mb_x, mb_y, origV_gpu, refV_gpu, V_COMPONENT, pixel_sads_gpu, block_sads_gpu, block_sads);
		}
	}

	cudaFree(pixel_sads_gpu);
	cudaFree(block_sads_gpu);
	free(block_sads);

	cudaFree(origY_gpu);
	cudaFree(origU_gpu);
	cudaFree(origV_gpu);

	cudaFree(refY_gpu);
	cudaFree(refU_gpu);
	cudaFree(refV_gpu);
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
