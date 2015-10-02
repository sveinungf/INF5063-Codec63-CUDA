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
void find_min(int k, int* values)
{
	int vals[9] = { 512, 256, 128, 64, 32, 16, 8, 4, 2 };

	for (int o = 0; o < 9; ++o) {
		int current = vals[o];

		if (k < current) {
			values[k] = min(values[k], values[k + current]);
		}

		__syncthreads();
	}

	if (k == 0) {
		values[0] = min(values[0], values[1]);
	}
}

__global__
void min_sad_block_index(uint8_t* orig_block, uint8_t* ref_search_range, int stride, int range_width, int range_height, int* index_result)
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

	find_min(k, block_sads);

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

	find_min(k, block_sads);

	__shared__ int index_result;

	if (k == 0) {
		index_result = INT_MAX;
	}

	__syncthreads();

	if (result == block_sads[0]) {
		atomicMin(&index_result, k);
	}

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

	uint8_t *origY_gpu, *origU_gpu, *origV_gpu;
	uint8_t *refY_gpu, *refU_gpu, *refV_gpu;

	const int frame_size_Y = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t);
	const int frame_size_U = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t);
	const int frame_size_V = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t);

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

	int* vectors_x = new int[cm->mb_rows * cm->mb_cols];
	int* vectors_y = new int[cm->mb_rows * cm->mb_cols];
	int w = cm->padw[Y_COMPONENT];
	int h = cm->padh[Y_COMPONENT];

	/* Luma */
	int *vector_x_gpu, *vector_y_gpu;
	const int vector_size = cm->mb_rows*cm->mb_cols*sizeof(int);

	cudaMalloc((void**) &vector_x_gpu, vector_size);
	cudaMalloc((void**) &vector_y_gpu, vector_size);

	int* lefts_gpu;
	int* rights_gpu;
	int* tops_gpu;
	int* bottoms_gpu;
	cudaMalloc((void**) &lefts_gpu, cm->mb_cols * sizeof(int));
	cudaMalloc((void**) &rights_gpu, cm->mb_cols * sizeof(int));
	cudaMalloc((void**) &tops_gpu, cm->mb_rows * sizeof(int));
	cudaMalloc((void**) &bottoms_gpu, cm->mb_rows * sizeof(int));

	int* lefts = new int[cm->mb_cols];
	int* rights = new int[cm->mb_cols];
	int* tops = new int[cm->mb_rows];
	int* bottoms = new int[cm->mb_rows];

	// LEFTS
	lefts[0] = 0;
	lefts[1] = 0;
	lefts[2] = 0;
	for (int i = 3; i < cm->mb_cols; ++i) {
		lefts[i] = (i-2)*8; // 2 pga range er 16
	}

	// RIGHTS
	for (int i = 0; i < cm->mb_cols-3; ++i) {
		rights[i] = (i+2)*8;
	}
	rights[cm->mb_cols-3] = w - 8;
	rights[cm->mb_cols-2] = w - 8;
	rights[cm->mb_cols-1] = w - 8;

	// TOPS
	tops[0] = 0;
	tops[1] = 0;
	tops[2] = 0;

	for (int i = 3; i < cm->mb_rows; ++i) {
		tops[i] = (i-2)*8;
	}

	// BOTTOMS
	for (int i = 0; i < cm->mb_rows-3; ++i) {
		bottoms[i] = (i+2)*8;
	}
	bottoms[cm->mb_rows-3] = h - 8;
	bottoms[cm->mb_rows-2] = h - 8;
	bottoms[cm->mb_rows-1] = h - 8;

	cudaMemcpy(lefts_gpu, lefts, cm->mb_cols * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(rights_gpu, rights, cm->mb_cols * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(tops_gpu, tops, cm->mb_rows * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(bottoms_gpu, bottoms, cm->mb_rows * sizeof(int), cudaMemcpyHostToDevice);

	dim3 numBlocks(cm->mb_cols, cm->mb_rows);
	dim3 threadsPerBlock(32, 32);
	me_block_8x8_gpu<<<numBlocks, threadsPerBlock>>>(origY_gpu, refY_gpu, lefts_gpu, rights_gpu, tops_gpu, bottoms_gpu, cm->me_search_range, w, h, vector_x_gpu, vector_y_gpu);

	delete[] lefts;
	delete[] rights;
	delete[] tops;
	delete[] bottoms;
	cudaFree(lefts_gpu);
	cudaFree(rights_gpu);
	cudaFree(tops_gpu);
	cudaFree(bottoms_gpu);

	cudaMemcpy(vectors_x, vector_x_gpu, vector_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(vectors_y, vector_y_gpu, vector_size, cudaMemcpyDeviceToHost);

	cudaFree(vector_x_gpu);
	cudaFree(vector_y_gpu);

	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			macroblock *mb = &cm->curframe->mbs[Y_COMPONENT][mb_y * w / 8 + mb_x];
			mb->mv_x = vectors_x[mb_y * cm->mb_cols + mb_x];
			mb->mv_y = vectors_y[mb_y * cm->mb_cols + mb_x];
			mb->use_mv = 1;
		}
	}

	delete[] vectors_x;
	delete[] vectors_y;

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			me_block_8x8(cm, mb_x, mb_y, origU_gpu, refU_gpu, U_COMPONENT);
			me_block_8x8(cm, mb_x, mb_y, origV_gpu, refV_gpu, V_COMPONENT);
		}
	}

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
