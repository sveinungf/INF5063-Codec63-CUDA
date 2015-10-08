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

template<int block_size>
__device__
static void min_reduce(int i, int* values)
{
	if (i < block_size/2)
	{
		// Intentionally no break between cases
		switch (block_size) {
			case 1024:
				values[i] = min(values[i], values[i + 512]);
				__syncthreads();
			case 512:
				values[i] = min(values[i], values[i + 256]);
				__syncthreads();
			case 256:
				values[i] = min(values[i], values[i + 128]);
				__syncthreads();
			case 128:
				values[i] = min(values[i], values[i + 64]);
				__syncthreads();
		}

		if (i < 32)
		{
			min_warp_reduce(i, values);
		}
	}
	else
	{
		switch (block_size) {
			case 1024:
				__syncthreads();
			case 512:
				__syncthreads();
			case 256:
				__syncthreads();
			case 128:
				__syncthreads();
		}
	}
}

template<int block_size>
__device__
static void first_min_occurrence(int i, int* values, int value, int* result)
{
	min_reduce<block_size>(i, values);

	if (i == 0) {
		*result = INT_MAX;
	}

	__syncthreads();

	if (value == values[0]) {
		atomicMin(result, i);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu(struct macroblock* mbs, uint8_t* orig, uint8_t* ref, int* lefts, int* rights, int* tops, int* bottoms, int w, int* vector_x, int* vector_y, int* use_mv)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int ref_mb_id = j*blockDim.x + i;

	const int mb_x = blockIdx.x;
	const int mb_y = blockIdx.y;
	const int orig_mb_id = mb_y*gridDim.x + mb_x;

	const int left = lefts[mb_x];
	const int top = tops[mb_y];
	const int right = rights[mb_x];
	const int bottom = bottoms[mb_y];

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	uint8_t* orig_block = orig + my * w + mx;
	uint8_t* ref_search_range = ref + top*w + left;

	__shared__ uint8_t shared_orig_block[64];

	if (i < 8 && j < 8)
	{
		shared_orig_block[j*8 + i] = orig_block[j*w + i];
	}

	__syncthreads();

	int block_sad = INT_MAX;

	const int range_width = right - left;
	const int range_height = bottom - top;

	const int shifts = (i % 4) * 8;

	// (i/4)*4 rounds i down to the nearest integer divisible by 4
	uint8_t* ref_block_top_row_aligned = ref_search_range + j*w + (i/4)*4;

	if (j < range_height && i < range_width)
	{
		block_sad = 0;

		for (int y = 0; y < 8; ++y)
		{
			uint32_t* ref_block_row_aligned = (uint32_t*) (ref_block_top_row_aligned + y*w);
			uint32_t ref_row_left = (ref_block_row_aligned[0] >> shifts) | (ref_block_row_aligned[1] << 32-shifts);
			uint32_t ref_row_right = (ref_block_row_aligned[1] >> shifts) | (ref_block_row_aligned[2] << 32-shifts);

			uint8_t* orig_block_row = shared_orig_block + y*8;
			uint32_t orig_row_left = *((uint32_t*) orig_block_row);
			uint32_t orig_row_right = *((uint32_t*) orig_block_row + 1);

			block_sad += __vsadu4(ref_row_left, orig_row_left);
			block_sad += __vsadu4(ref_row_right, orig_row_right);
		}
	}

	const int max_range_width = range * 2;
	const int max_range_height = range * 2;
	const int max_mb_count = max_range_width * max_range_height;

	__shared__ int block_sads[max_mb_count];

	block_sads[ref_mb_id] = block_sad;

	__syncthreads();

	__shared__ int index_result;
	first_min_occurrence<max_mb_count>(ref_mb_id, block_sads, block_sad, &index_result);

	__syncthreads();

	if (ref_mb_id == 0)
	{
		/* Here, there should be a threshold on SAD that checks if the motion vector
		     is cheaper than intraprediction. We always assume MV to be beneficial */
		struct macroblock* mb = &mbs[orig_mb_id];
		mb->use_mv = 1;
		mb->mv_x = left + (index_result % max_range_width) - mx;
		mb->mv_y = top + (index_result / max_range_width) - my;
	}
}

void c63_motion_estimate(struct c63_common *cm)
{
	struct macroblock** mbs = cm->curframe->mbs_gpu;
	yuv_t* orig = cm->curframe->orig_gpu;
	yuv_t* ref = cm->refframe->recons_gpu;

	const int frame_size_Y = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT] * sizeof(uint8_t);
	const int frame_size_U = cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT] * sizeof(uint8_t);
	const int frame_size_V = cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT] * sizeof(uint8_t);

	cudaMemcpy(ref->Y, cm->refframe->recons->Y, frame_size_Y, cudaMemcpyHostToDevice);
	cudaMemcpy(ref->U, cm->refframe->recons->U, frame_size_U, cudaMemcpyHostToDevice);
	cudaMemcpy(ref->V, cm->refframe->recons->V, frame_size_V, cudaMemcpyHostToDevice);

	const int wY = cm->padw[Y_COMPONENT];
	const int wU = cm->padw[U_COMPONENT];
	const int wV = cm->padw[V_COMPONENT];

	/* Luma */
	dim3 numBlocksY(cm->mb_colsY, cm->mb_rowsY);
	dim3 threadsPerBlockY(ME_RANGE_Y*2, ME_RANGE_Y*2);
	me_block_8x8_gpu<ME_RANGE_Y><<<numBlocksY, threadsPerBlockY>>>(mbs[Y_COMPONENT], orig->Y, ref->Y, cm->cuda_me.leftsY_gpu, cm->cuda_me.rightsY_gpu, cm->cuda_me.topsY_gpu, cm->cuda_me.bottomsY_gpu, wY, cm->cuda_me.vector_xY_gpu, cm->cuda_me.vector_yY_gpu, cm->cuda_me.use_mvY_gpu);
	cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], mbs[Y_COMPONENT], cm->mb_rowsY * cm->mb_colsY * sizeof(struct macroblock), cudaMemcpyDeviceToHost);

	/* Chroma */
	dim3 numBlocksUV(cm->mb_colsUV, cm->mb_rowsUV);
	dim3 threadsPerBlockUV(ME_RANGE_UV*2, ME_RANGE_UV*2);
	me_block_8x8_gpu<ME_RANGE_UV><<<numBlocksUV, threadsPerBlockUV>>>(mbs[U_COMPONENT], orig->U, ref->U, cm->cuda_me.leftsUV_gpu, cm->cuda_me.rightsUV_gpu, cm->cuda_me.topsUV_gpu, cm->cuda_me.bottomsUV_gpu, wU, cm->cuda_me.vector_xU_gpu, cm->cuda_me.vector_yU_gpu, cm->cuda_me.use_mvU_gpu);
	cudaMemcpy(cm->curframe->mbs[U_COMPONENT], mbs[U_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock), cudaMemcpyDeviceToHost);

	me_block_8x8_gpu<ME_RANGE_UV><<<numBlocksUV, threadsPerBlockUV>>>(mbs[V_COMPONENT], orig->V, ref->V, cm->cuda_me.leftsUV_gpu, cm->cuda_me.rightsUV_gpu, cm->cuda_me.topsUV_gpu, cm->cuda_me.bottomsUV_gpu, wV, cm->cuda_me.vector_xV_gpu, cm->cuda_me.vector_yV_gpu, cm->cuda_me.use_mvV_gpu);
	cudaMemcpy(cm->curframe->mbs[V_COMPONENT], mbs[V_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock), cudaMemcpyDeviceToHost);
}

/* Motion compensation for 8x8 block */
__global__
static void mc_block_8x8_gpu(struct macroblock* mbs, int w, uint8_t *predicted, uint8_t *ref, int* vector_x, int* vector_y, int* use_mv)
{
	int mb_x = blockIdx.x;
	int mb_y = threadIdx.x;

	int index = mb_y * w / 8 + mb_x;

	struct macroblock* mb = &mbs[index];

	if (!mb->use_mv) {
		return;
	}

	int mv_x = mb->mv_x;
	int mv_y = mb->mv_y;

	int left = mb_x * 8;
	int top = mb_y * 8;
	int right = left + 8;
	int bottom = top + 8;

	/* Copy block from ref mandated by MV */
	int x, y;

	for (y = top; y < bottom; ++y)
	{
		for (x = left; x < right; ++x)
		{
			predicted[y * w + x] = ref[(y + mv_y) * w + (x + mv_x)];
		}
	}
}

void c63_motion_compensate(struct c63_common *cm)
{
	struct macroblock** mbs = cm->curframe->mbs_gpu;
	yuv_t* ref = cm->refframe->recons_gpu;

	int wY = cm->padw[Y_COMPONENT];
	int wU = cm->padw[U_COMPONENT];
	int wV = cm->padw[V_COMPONENT];

	/* Luma */
	// TODO: Number of macroblock rows are now limited to 1024. Number of threads per block should
	// ideally be a multiplum of the warp size (32).
	mc_block_8x8_gpu<<<cm->mb_colsY, cm->mb_rowsY>>>(mbs[Y_COMPONENT], wY, cm->cuda_me.predY_gpu, ref->Y, cm->cuda_me.vector_xY_gpu, cm->cuda_me.vector_yY_gpu, cm->cuda_me.use_mvY_gpu);
	cudaMemcpy(cm->curframe->predicted->Y, cm->cuda_me.predY_gpu, cm->ypw * cm->yph * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	/* Chroma */
	mc_block_8x8_gpu<<<cm->mb_colsUV, cm->mb_rowsUV>>>(mbs[U_COMPONENT], wU, cm->cuda_me.predU_gpu, ref->U, cm->cuda_me.vector_xU_gpu, cm->cuda_me.vector_yU_gpu, cm->cuda_me.use_mvU_gpu);
	cudaMemcpy(cm->curframe->predicted->U, cm->cuda_me.predU_gpu, cm->upw * cm->uph * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	mc_block_8x8_gpu<<<cm->mb_colsUV, cm->mb_rowsUV>>>(mbs[V_COMPONENT], wV, cm->cuda_me.predV_gpu, ref->V, cm->cuda_me.vector_xV_gpu, cm->cuda_me.vector_yV_gpu, cm->cuda_me.use_mvV_gpu);
	cudaMemcpy(cm->curframe->predicted->V, cm->cuda_me.predV_gpu, cm->vpw * cm->vph * sizeof(uint8_t), cudaMemcpyDeviceToHost);
}
