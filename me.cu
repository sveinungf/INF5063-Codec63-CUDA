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

template<int range>
__global__
static void me_block_8x8_gpu_Y(const uint8_t* __restrict__ orig, const uint8_t* __restrict__ ref,
		const int* __restrict__ lefts, const int* __restrict__ rights, const int* __restrict__ tops,
		const int* __restrict__ bottoms, int w, unsigned int* __restrict__ index_results)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int tid = j*blockDim.x + i;
	const int ref_mb_id = j*4*blockDim.x + i;
	const int ref_mb2_id = (j*4+1)*blockDim.x + i;
	const int ref_mb3_id = (j*4+2)*blockDim.x + i;
	const int ref_mb4_id = (j*4+3)*blockDim.x + i;

	const int mb_x = blockIdx.x;
	const int mb_y = blockIdx.y;
	const int orig_mb_id = mb_y*gridDim.x + mb_x;

	const int left = lefts[mb_x];
	const int top = tops[mb_y];
	const int right = rights[mb_x];
	const int bottom = bottoms[mb_y];

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	const uint8_t* orig_block = orig + my * w + mx;
	const uint8_t* ref_search_range = ref + top*w + left;

	__shared__ uint8_t shared_orig_block[64];

	if (i < 8 && j < 8)
	{
		shared_orig_block[j*8 + i] = orig_block[j*w + i];
	}

	__syncthreads();

	int block_sad = INT_MAX;
	int block2_sad = INT_MAX;
	int block3_sad = INT_MAX;
	int block4_sad = INT_MAX;

	const int range_width = right - left;
	const int range_height = (bottom - top)/4;

	const unsigned int mask = 0x3210 + 0x1111 * (i%4);

	// (i/4)*4 rounds i down to the nearest integer divisible by 4
	const uint8_t* ref_block_top_row_aligned = ref_search_range + (j*4)*w + (i/4)*4;

	if (j < range_height && i < range_width)
	{
		block_sad = 0;
		block2_sad = 0;
		block3_sad = 0;
		block4_sad = 0;

		#pragma unroll
		for (int y = 0; y < 8; ++y)
		{
			uint32_t* ref_block_row_aligned = (uint32_t*) (ref_block_top_row_aligned + y*w);
			uint32_t ref_row_left = __byte_perm(ref_block_row_aligned[0], ref_block_row_aligned[1], mask);
			uint32_t ref_row_right = __byte_perm(ref_block_row_aligned[1], ref_block_row_aligned[2], mask);

			uint32_t* ref_block2_row_aligned = (uint32_t*) (ref_block_top_row_aligned + (y+1)*w);
			uint32_t ref_row2_left = __byte_perm(ref_block2_row_aligned[0], ref_block2_row_aligned[1], mask);
			uint32_t ref_row2_right = __byte_perm(ref_block2_row_aligned[1], ref_block2_row_aligned[2], mask);

			uint32_t* ref_block3_row_aligned = (uint32_t*) (ref_block_top_row_aligned + (y+2)*w);
			uint32_t ref_row3_left = __byte_perm(ref_block3_row_aligned[0], ref_block3_row_aligned[1], mask);
			uint32_t ref_row3_right = __byte_perm(ref_block3_row_aligned[1], ref_block3_row_aligned[2], mask);

			uint32_t* ref_block4_row_aligned = (uint32_t*) (ref_block_top_row_aligned + (y+3)*w);
			uint32_t ref_row4_left = __byte_perm(ref_block4_row_aligned[0], ref_block4_row_aligned[1], mask);
			uint32_t ref_row4_right = __byte_perm(ref_block4_row_aligned[1], ref_block4_row_aligned[2], mask);

			uint8_t* orig_block_row = shared_orig_block + y*8;
			uint32_t orig_row_left = *((uint32_t*) orig_block_row);
			uint32_t orig_row_right = *((uint32_t*) orig_block_row + 1);

			block_sad += __vsadu4(ref_row_left, orig_row_left);
			block_sad += __vsadu4(ref_row_right, orig_row_right);

			block2_sad += __vsadu4(ref_row2_left, orig_row_left);
			block2_sad += __vsadu4(ref_row2_right, orig_row_right);

			block3_sad += __vsadu4(ref_row3_left, orig_row_left);
			block3_sad += __vsadu4(ref_row3_right, orig_row_right);

			block4_sad += __vsadu4(ref_row4_left, orig_row_left);
			block4_sad += __vsadu4(ref_row4_right, orig_row_right);
		}
	}

	__shared__ int block_sads[32*32];

	block_sads[ref_mb_id] = block_sad;
	block_sads[ref_mb2_id] = block2_sad;
	block_sads[ref_mb3_id] = block3_sad;
	block_sads[ref_mb4_id] = block4_sad;

	__syncthreads();

	block_sads[tid] = min(block_sads[tid], block_sads[tid + 512]);
	block_sads[tid + 256] = min(block_sads[tid + 256], block_sads[tid + 768]);
	__syncthreads();

	block_sads[tid] = min(block_sads[tid], block_sads[tid + 256]);
	__syncthreads();

	if (tid < 128) {
		block_sads[tid] = min(block_sads[tid], block_sads[tid + 128]);
	}
	__syncthreads();

	if (tid < 64) {
		block_sads[tid] = min(block_sads[tid], block_sads[tid + 64]);
	}
	__syncthreads();

	if (tid < 32) {
		min_warp_reduce(tid, block_sads);
	}
	__syncthreads();

	int min = block_sads[0];

	if (block_sad == min) {
		atomicMin(index_results + orig_mb_id, ref_mb_id);
	}

	if (block2_sad == min) {
		atomicMin(index_results + orig_mb_id, ref_mb2_id);
	}

	if (block3_sad == min) {
		atomicMin(index_results + orig_mb_id, ref_mb3_id);
	}

	if (block4_sad == min) {
		atomicMin(index_results + orig_mb_id, ref_mb4_id);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu_UV(const uint8_t* __restrict__ orig, const uint8_t* __restrict__ ref,
		const int* __restrict__ lefts, const int* __restrict__ rights, const int* __restrict__ tops,
		const int* __restrict__ bottoms, int w, unsigned int* __restrict__ index_results)
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

	const uint8_t* orig_block = orig + my * w + mx;
	const uint8_t* ref_search_range = ref + top*w + left;

	__shared__ uint8_t shared_orig_block[64];

	if (i < 8 && j < 8)
	{
		shared_orig_block[j*8 + i] = orig_block[j*w + i];
	}

	__syncthreads();

	int block_sad = INT_MAX;

	const int range_width = right - left;
	const int range_height = bottom - top;

	const unsigned int mask = 0x3210 + 0x1111 * (i%4);

	// (i/4)*4 rounds i down to the nearest integer divisible by 4
	const uint8_t* ref_block_top_row_aligned = ref_search_range + j*w + (i/4)*4;

	if (j < range_height && i < range_width)
	{
		block_sad = 0;

		#pragma unroll
		for (unsigned int y = 8; y--; )
		{
			uint32_t* ref_block_row_aligned = (uint32_t*) (ref_block_top_row_aligned + y*w);
			uint32_t ref_row_left = __byte_perm(ref_block_row_aligned[0], ref_block_row_aligned[1], mask);
			uint32_t ref_row_right = __byte_perm(ref_block_row_aligned[1], ref_block_row_aligned[2], mask);

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

	min_reduce<max_mb_count>(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, ref_mb_id);
	}
}

template<int range>
__global__
static void set_motion_vectors(struct macroblock* __restrict__ mbs, const int* __restrict__ lefts,
		int* tops, const unsigned int* __restrict__ index_results)
{
	const int mb_x = blockIdx.x;
	const int mb_y = threadIdx.x;
	const int orig_mb_id = mb_y*gridDim.x + mb_x;

	const int left = lefts[mb_x];
	const int top = tops[mb_y];

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	int index_result = index_results[orig_mb_id];

	/* Here, there should be a threshold on SAD that checks if the motion vector
		 is cheaper than intraprediction. We always assume MV to be beneficial */
	struct macroblock* mb = &mbs[orig_mb_id];
	mb->use_mv = 1;
	mb->mv_x = left + (index_result % (range*2)) - mx;
	mb->mv_y = top + (index_result / (range*2)) - my;
}

void c63_motion_estimate(struct c63_common *cm)
{
	struct macroblock** mbs = cm->curframe->mbs_gpu;
	yuv_t* orig = cm->curframe->orig_gpu;
	yuv_t* ref = cm->refframe->recons_gpu;

	const int wY = cm->padw[Y_COMPONENT];
	const int wU = cm->padw[U_COMPONENT];
	const int wV = cm->padw[V_COMPONENT];

	/* Luma */
	dim3 numBlocksY(cm->mb_colsY, cm->mb_rowsY);
	dim3 threadsPerBlockY(ME_RANGE_Y*2, ME_RANGE_Y/2);

	cudaMemsetAsync(cm->cuda_data.sad_index_resultsY, 255, cm->mb_colsY*cm->mb_rowsY*sizeof(unsigned int), cm->cuda_data.streamY);
	me_block_8x8_gpu_Y<ME_RANGE_Y><<<numBlocksY, threadsPerBlockY, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.leftsY_gpu, cm->cuda_data.rightsY_gpu, cm->cuda_data.topsY_gpu, cm->cuda_data.bottomsY_gpu, wY, cm->cuda_data.sad_index_resultsY);
	set_motion_vectors<ME_RANGE_Y><<<cm->mb_colsY, cm->mb_rowsY, 0, cm->cuda_data.streamY>>>(mbs[Y_COMPONENT], cm->cuda_data.leftsY_gpu, cm->cuda_data.topsY_gpu, cm->cuda_data.sad_index_resultsY);
	cudaMemcpyAsync(cm->curframe->mbs[Y_COMPONENT], mbs[Y_COMPONENT], cm->mb_rowsY * cm->mb_colsY * sizeof(struct macroblock), cudaMemcpyDeviceToHost, cm->cuda_data.streamY);

	/* Chroma */
	dim3 numBlocksUV(cm->mb_colsUV, cm->mb_rowsUV);
	dim3 threadsPerBlockUV(ME_RANGE_UV*2, ME_RANGE_UV*2);

	cudaMemsetAsync(cm->cuda_data.sad_index_resultsU, 255, cm->mb_colsUV*cm->mb_rowsUV*sizeof(unsigned int), cm->cuda_data.streamU);
	me_block_8x8_gpu_UV<ME_RANGE_UV><<<numBlocksUV, threadsPerBlockUV, 0, cm->cuda_data.streamU>>>(orig->U, ref->U, cm->cuda_data.leftsUV_gpu, cm->cuda_data.rightsUV_gpu, cm->cuda_data.topsUV_gpu, cm->cuda_data.bottomsUV_gpu, wU, cm->cuda_data.sad_index_resultsU);
	set_motion_vectors<ME_RANGE_UV><<<cm->mb_colsUV, cm->mb_rowsUV, 0, cm->cuda_data.streamU>>>(mbs[U_COMPONENT], cm->cuda_data.leftsUV_gpu, cm->cuda_data.topsUV_gpu, cm->cuda_data.sad_index_resultsU);
	cudaMemcpyAsync(cm->curframe->mbs[U_COMPONENT], mbs[U_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock), cudaMemcpyDeviceToHost, cm->cuda_data.streamU);

	cudaMemsetAsync(cm->cuda_data.sad_index_resultsV, 255, cm->mb_colsUV*cm->mb_rowsUV*sizeof(unsigned int), cm->cuda_data.streamV);
	me_block_8x8_gpu_UV<ME_RANGE_UV><<<numBlocksUV, threadsPerBlockUV, 0, cm->cuda_data.streamV>>>(orig->V, ref->V, cm->cuda_data.leftsUV_gpu, cm->cuda_data.rightsUV_gpu, cm->cuda_data.topsUV_gpu, cm->cuda_data.bottomsUV_gpu, wV, cm->cuda_data.sad_index_resultsV);
	set_motion_vectors<ME_RANGE_UV><<<cm->mb_colsUV, cm->mb_rowsUV, 0, cm->cuda_data.streamV>>>(mbs[V_COMPONENT], cm->cuda_data.leftsUV_gpu, cm->cuda_data.topsUV_gpu, cm->cuda_data.sad_index_resultsV);
	cudaMemcpyAsync(cm->curframe->mbs[V_COMPONENT], mbs[V_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock), cudaMemcpyDeviceToHost, cm->cuda_data.streamV);
}

/* Motion compensation for 8x8 block */
__global__
static void mc_block_8x8_gpu(const struct macroblock* __restrict__ mbs, int w,
		uint8_t* __restrict__ predicted, const uint8_t* __restrict__ ref)
{
	int mb_x = blockIdx.x;
	int mb_y = threadIdx.x;

	int index = mb_y * w / 8 + mb_x;

	const struct macroblock* mb = &mbs[index];

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
	yuv_t* pred = cm->curframe->predicted_gpu;
	yuv_t* ref = cm->refframe->recons_gpu;

	int wY = cm->padw[Y_COMPONENT];
	int wU = cm->padw[U_COMPONENT];
	int wV = cm->padw[V_COMPONENT];

	/* Luma */
	// TODO: Number of macroblock rows are now limited to 1024. Number of threads per block should
	// ideally be a multiplum of the warp size (32).
	mc_block_8x8_gpu<<<cm->mb_colsY, cm->mb_rowsY, 0, cm->cuda_data.streamY>>>(mbs[Y_COMPONENT], wY, pred->Y, ref->Y);

	/* Chroma */
	mc_block_8x8_gpu<<<cm->mb_colsUV, cm->mb_rowsUV, 0, cm->cuda_data.streamU>>>(mbs[U_COMPONENT], wU, pred->U, ref->U);
	mc_block_8x8_gpu<<<cm->mb_colsUV, cm->mb_rowsUV, 0, cm->cuda_data.streamV>>>(mbs[V_COMPONENT], wV, pred->V, ref->V);
}
