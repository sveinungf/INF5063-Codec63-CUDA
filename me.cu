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


__device__
static void min_warp_reduce(int i, volatile int* values)
{
	values[i] = min(values[i], values[i + 32]);
	values[i] = min(values[i], values[i + 16]);
	values[i] = min(values[i], values[i + 8]);
	values[i] = min(values[i], values[i + 4]);
	values[i] = min(values[i], values[i + 2]);
	values[i] = min(values[i], values[i + 1]);
}

__device__
static void min_reduce(int i, int* values)
{
	const int initial_stride = (blockDim.x * blockDim.y) / 2;

	if (i < 512)
	{
		for (int stride = initial_stride; stride > 32; stride >>= 1)
		{
			values[i] = min(values[i], values[i + stride]);
			__syncthreads();
		}

		if (i < 32)
		{
			min_warp_reduce(i, values);
		}
	}
	else
	{
		for (int stride = initial_stride; stride > 32; stride >>= 1)
		{
			__syncthreads();
		}
	}
}

__device__
static void first_min_occurrence(int i, int* values, int value, int* result)
{
	min_reduce(i, values);

	if (i == 0) {
		*result = INT_MAX;
	}

	__syncthreads();

	if (value == values[0]) {
		atomicMin(result, i);
	}
}

__global__
static void min_sad_block_index(uint8_t* orig_block, uint8_t* ref_search_range, int stride, int range_width, int range_height, int* index_result)
{
	int i = threadIdx.x;
	int j = threadIdx.y;

	__shared__ uint8_t shared_orig_block[64];

	if (i < 8 && j < 8)
	{
		shared_orig_block[i*8 + j] = orig_block[i*stride + j];
	}

	__syncthreads();

	int result;

	if (j < range_height && i < range_width)
	{
		uint8_t* ref_block = ref_search_range + j*stride + i;
		result = 0;

		for (int y = 0; y < 8; ++y)
		{
			for (int x = 0; x < 8; ++x)
			{
				result += abs(ref_block[y*stride + x] - shared_orig_block[y*8 + x]);
			}
		}
	}
	else
	{
		result = INT_MAX;
	}

	__shared__ int block_sads[1024];

	int k = j*blockDim.x + i;
	block_sads[k] = result;

	__syncthreads();

	min_reduce(k, block_sads);

	if (k == 0) {
		*index_result = INT_MAX;
	}

	__syncthreads();

	if (result == block_sads[0]) {
		atomicMin(index_result, k);
	}
}

__global__
static void me_block_8x8_gpu(uint8_t* orig_gpu, uint8_t* ref_gpu, int* lefts, int* rights, int* tops, int* bottoms, int range, int w, int h, int* vector_x, int* vector_y)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int k = j*blockDim.x + i;

	const int mb_x = blockIdx.x;
	const int mb_y = blockIdx.y;

	int left = lefts[mb_x];
	int top = tops[mb_y];
	int right = rights[mb_x];
	int bottom = bottoms[mb_y];

	int mx = mb_x * 8;
	int my = mb_y * 8;

	int range_width = right - left;
	int range_height = bottom - top;

	uint8_t* orig_block = orig_gpu + my * w + mx;
	uint8_t* ref_search_range = ref_gpu + top*w + left;

	__shared__ uint8_t shared_orig_block[64];

	if (i < 8 && j < 8)
	{
		shared_orig_block[i*8 + j] = orig_block[i*w + j];
	}

	__syncthreads();

	int result;

	if (j < range_height && i < range_width)
	{
		uint8_t* ref_block = ref_search_range + j*w + i;
		result = 0;

		for (int y = 0; y < 8; ++y)
		{
			for (int x = 0; x < 8; ++x)
			{
				result += abs(ref_block[y*w + x] - shared_orig_block[y*8 + x]);
			}
		}
	}
	else
	{
		result = INT_MAX;
	}

	__shared__ int block_sads[1024];

	block_sads[k] = result;

	__syncthreads();

	__shared__ int index_result;
	first_min_occurrence(k, block_sads, result, &index_result);

	__syncthreads();

	if (k == 0)
	{
		vector_x[blockIdx.y*gridDim.x + blockIdx.x] = left + (index_result%32) - mx;
		vector_y[blockIdx.y*gridDim.x + blockIdx.x] = top + (index_result/32) - my;
	}
}

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *orig_gpu, uint8_t *ref_gpu, int color_component)
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

	int range_width = right - left;
	int range_height = bottom - top;

	uint8_t* orig_block_gpu = orig_gpu + my * w + mx;
	uint8_t* ref_search_range_gpu = ref_gpu + top*w + left;

	int result;

	int* result_gpu;
	cudaMalloc((void**) &result_gpu, sizeof(int));

	int numBlocks = 1;
	dim3 threadsPerBlock(32, 32);
	min_sad_block_index<<<numBlocks, threadsPerBlock>>>(orig_block_gpu, ref_search_range_gpu, w, range_width, range_height, result_gpu);
	cudaMemcpy(&result, result_gpu, sizeof(int), cudaMemcpyDeviceToHost);

	mb->mv_x = left + (result%32) - mx;
	mb->mv_y = top + (result/32) - my;

	cudaFree(result_gpu);

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

	int wY = cm->padw[Y_COMPONENT];
	int hY = cm->padh[Y_COMPONENT];

	/* Luma */
	dim3 numBlocksY(cm->mb_cols, cm->mb_rows);
	dim3 threadsPerBlockY(32, 32);
	me_block_8x8_gpu<<<numBlocksY, threadsPerBlockY>>>(cm->cuda_me.origY_gpu, cm->cuda_me.refY_gpu, cm->cuda_me.lefts_gpu, cm->cuda_me.rightsY_gpu, cm->cuda_me.tops_gpu, cm->cuda_me.bottomsY_gpu, cm->me_search_range, wY, hY, cm->cuda_me.vector_x_gpu, cm->cuda_me.vector_y_gpu);

	const int vector_sizeY = cm->mb_rows*cm->mb_cols*sizeof(int);
	cudaMemcpy(cm->cuda_me.vector_x, cm->cuda_me.vector_x_gpu, vector_sizeY, cudaMemcpyDeviceToHost);
	cudaMemcpy(cm->cuda_me.vector_y, cm->cuda_me.vector_y_gpu, vector_sizeY, cudaMemcpyDeviceToHost);

	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			macroblock *mb = &cm->curframe->mbs[Y_COMPONENT][mb_y * wY / 8 + mb_x];
			mb->mv_x = cm->cuda_me.vector_x[mb_y * cm->mb_cols + mb_x];
			mb->mv_y = cm->cuda_me.vector_y[mb_y * cm->mb_cols + mb_x];
			mb->use_mv = 1;
		}
	}

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			me_block_8x8(cm, mb_x, mb_y, cm->cuda_me.origU_gpu, cm->cuda_me.refU_gpu, U_COMPONENT);
			me_block_8x8(cm, mb_x, mb_y, cm->cuda_me.origV_gpu, cm->cuda_me.refV_gpu, V_COMPONENT);
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
