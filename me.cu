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
	/*if (i < block_size/2)
	{*/
		// Intentionally no break between cases
		switch (block_size) {
			case 1024:
				if (i < 512)
					values[i] = min(values[i], values[i + 512]);
				__syncthreads();
			case 512:
				if (i < 256)
					values[i] = min(values[i], values[i + 256]);
				__syncthreads();
			case 256:
				if (i < 128)
					values[i] = min(values[i], values[i + 128]);
				__syncthreads();
			case 128:
				if (i < 64)
					values[i] = min(values[i], values[i + 64]);
				__syncthreads();
		}

		if (i < 32)
		{
			min_warp_reduce(i, values);
		}
	/*}
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
	}*/
}

__device__
static void min_reduce384(int i, int* values)
{
	if (i < 128)
	{
		values[i + 128] = min(values[i + 128], values[i + 256]);
	}

	__syncthreads();

	min_reduce<256>(i, values);
}

__device__
static void min_reduce768(int i, int* values)
{
	if (i < 256)
	{
		values[i + 256] = min(values[i + 256], values[i + 512]);
	}

	__syncthreads();

	min_reduce<512>(i, values);
}

__device__
static void min_reduce576(int i, int* values)
{
	if (i < 64)
	{
		values[i + 448] = min(values[i + 448], values[i + 512]);
	}

	__syncthreads();

	min_reduce<512>(i, values);
}

template<int range_w, int range_h>
__device__
static int calculate_block_sad_new(uint8_t* orig_block, uint8_t* ref_search_range, int w)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;

	__shared__ uint8_t shared_orig_block[64];

	const int actual_width = range_w + 7;
	const int actual_height = range_h + 7;

	const int aligned_ref_width = (actual_width + 6)*4; // 3 before and 3 after, not really used
	__shared__ uint8_t aligned_ref[actual_height * aligned_ref_width];
	uint8_t* start = aligned_ref + 3*4; // skip the 3 before

	const int actual_width_half = (range_w + 8) / 2; // technically not really half
	const int actual_height_half = (range_h + 8) / 2;

	if (range_w < 24 || range_h < 24)
	{
		if (j < 8 && i < 8)
		{
			shared_orig_block[j*8 + i] = orig_block[j*w + i];
		}
	}

	if (j < actual_height_half && i < actual_width_half) { // (32+8) / 2 = 20
		int col_left = i;
		int col_right = i + (actual_width_half - 1); // 19 so we don't risk reading outside the allocated memory
		int row_top = j;
		int row_bottom = j + (actual_height_half - 1);

		// Top left
		uint8_t val = ref_search_range[row_top*w + col_left];
		start[row_top*aligned_ref_width + (col_left*4)] = val;
		start[row_top*aligned_ref_width + (col_left*4)-3] = val;
		start[row_top*aligned_ref_width + (col_left*4)-6] = val;
		start[row_top*aligned_ref_width + (col_left*4)-9] = val;

		// Top right
		val = ref_search_range[j*w + col_right];
		start[row_top*aligned_ref_width + (col_right*4)] = val;
		start[row_top*aligned_ref_width + (col_right*4)-3] = val;
		start[row_top*aligned_ref_width + (col_right*4)-6] = val;
		start[row_top*aligned_ref_width + (col_right*4)-9] = val;

		// Bottom left
		val = ref_search_range[row_bottom*w + col_left];
		start[row_bottom*aligned_ref_width + (col_left*4)] = val;
		start[row_bottom*aligned_ref_width + (col_left*4)-3] = val;
		start[row_bottom*aligned_ref_width + (col_left*4)-6] = val;
		start[row_bottom*aligned_ref_width + (col_left*4)-9] = val;

		// Bottom right
		val = ref_search_range[row_bottom*w + col_right];
		start[row_bottom*aligned_ref_width + (col_right*4)] = val;
		start[row_bottom*aligned_ref_width + (col_right*4)-3] = val;
		start[row_bottom*aligned_ref_width + (col_right*4)-6] = val;
		start[row_bottom*aligned_ref_width + (col_right*4)-9] = val;
	}
	else if (range_w >= 24 && range_h >= 24)
	{
		if (j >= actual_height_half && j < (actual_height_half + 8) && i >= actual_width_half && i < (actual_width_half + 8))
		{
			int col = i - actual_width_half;
			int row = j - actual_height_half;
			shared_orig_block[row*8 + col] = orig_block[row*w + col];
		}
	}

	__syncthreads();

	int result = 0;

	for (int y = 0; y < 8; ++y) {
		uint32_t* a1 = (uint32_t*) (start + (j+y)*aligned_ref_width) + i;
		uint32_t* a2 = a1 + 4;

		uint32_t* b1 = (uint32_t*) (shared_orig_block + y*8);
		uint32_t* b2 = b1 + 1;

		result += __vsadu4(*a1, *b1);
		result += __vsadu4(*a2, *b2);
	}

	return result;
}

template<int range_w, int range_h>
__device__
static int calculate_block_sad_old(uint8_t* orig_block, uint8_t* ref_search_range, int w)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;

	__shared__ uint8_t shared_orig_block[64];

	if (i < 8 && j < 8)
	{
		shared_orig_block[j*8 + i] = orig_block[j*w + i];
	}

	__syncthreads();

	int block_sad = INT_MAX;

	const int shifts = (i % 4) * 8;

	// (i/4)*4 rounds i down to the nearest integer divisible by 4
	uint8_t* ref_block_top_row_aligned = ref_search_range + j*w + (i/4)*4;

	int result = 0;

	for (int y = 0; y < 8; ++y)
	{
		uint32_t* ref_block_row_aligned = (uint32_t*) (ref_block_top_row_aligned + y*w);
		uint32_t ref_row_left = (ref_block_row_aligned[0] >> shifts) | (ref_block_row_aligned[1] << 32-shifts);
		uint32_t ref_row_right = (ref_block_row_aligned[1] >> shifts) | (ref_block_row_aligned[2] << 32-shifts);

		uint8_t* orig_block_row = shared_orig_block + y*8;
		uint32_t orig_row_left = *((uint32_t*) orig_block_row);
		uint32_t orig_row_right = *((uint32_t*) orig_block_row + 1);

		result += __vsadu4(ref_row_left, orig_row_left);
		result += __vsadu4(ref_row_right, orig_row_right);
	}

	return result;
}

template<int range>
__global__
static void me_block_8x8_gpu(struct macroblock* mbs, uint8_t* orig, uint8_t* ref, int* lefts, int* rights, int* tops, int* bottoms, int w, unsigned int* index_results)
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

	min_reduce<max_mb_count>(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, ref_mb_id);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu_Y_outer_horizontal(uint8_t* orig, uint8_t* ref, struct boundaries boundaries, int cols, int rows, int w, unsigned int* index_results)
{
	const int mb_ys[2] = { 0, rows-1 };

	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int ref_mb_id = j*blockDim.x + i;

	const int mb_x = blockIdx.x + 2;
	const int mb_y = mb_ys[blockIdx.y];
	const int orig_mb_id = mb_y*cols + mb_x;

	const int left = boundaries.lefts[mb_x];
	const int top = boundaries.tops[mb_y];
	const int right = boundaries.rights[mb_x];
	const int bottom = boundaries.bottoms[mb_y];

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	uint8_t* orig_block = orig + my * w + mx;
	uint8_t* ref_search_range = ref + top*w + left;

	__shared__ int block_sads[16*32];

	int block_sad = calculate_block_sad_new<32,16>(orig_block, ref_search_range, w);

	block_sads[ref_mb_id] = block_sad;

	__syncthreads();

	min_reduce<16*32>(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, j*32+i);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu_Y_inner_horizontal(uint8_t* orig, uint8_t* ref, struct boundaries boundaries, int cols, int rows, int w, unsigned int* index_results)
{
	const int mb_ys[2] = { 1, rows-2 };

	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int ref_mb_id = j*blockDim.x + i;

	const int mb_x = blockIdx.x + 2;
	const int mb_y = mb_ys[blockIdx.y];
	const int orig_mb_id = mb_y*cols + mb_x;

	const int left = boundaries.lefts[mb_x];
	const int top = boundaries.tops[mb_y];
	const int right = boundaries.rights[mb_x];
	const int bottom = boundaries.bottoms[mb_y];

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	uint8_t* orig_block = orig + my * w + mx;
	uint8_t* ref_search_range = ref + top*w + left;

	__shared__ int block_sads[24*32];

	int block_sad = calculate_block_sad_new<32,24>(orig_block, ref_search_range, w);

	block_sads[ref_mb_id] = block_sad;

	__syncthreads();

	min_reduce768(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, j*32+i);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu_Y_outer_vertical(uint8_t* orig, uint8_t* ref, struct boundaries boundaries, int cols, int rows, int w, unsigned int* index_results)
{
	const int mb_xs[2] = { 0, cols-1 };

	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int ref_mb_id = j*blockDim.x + i;

	const int mb_x = mb_xs[blockIdx.y];
	const int mb_y = blockIdx.x + 2;
	const int orig_mb_id = mb_y*cols + mb_x;

	const int left = boundaries.lefts[mb_x];
	const int top = boundaries.tops[mb_y];
	const int right = boundaries.rights[mb_x];
	const int bottom = boundaries.bottoms[mb_y];

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	uint8_t* orig_block = orig + my * w + mx;
	uint8_t* ref_search_range = ref + top*w + left;

	__shared__ int block_sads[16*32];

	int block_sad = calculate_block_sad_new<16,32>(orig_block, ref_search_range, w);

	block_sads[ref_mb_id] = block_sad;

	__syncthreads();

	min_reduce<16*32>(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, j*32+i);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu_Y_inner_vertical(uint8_t* orig, uint8_t* ref, struct boundaries boundaries, int cols, int rows, int w, unsigned int* index_results)
{
	const int mb_xs[2] = { 1, cols-2 };

	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int ref_mb_id = j*blockDim.x + i;

	const int mb_x = mb_xs[blockIdx.y];
	const int mb_y = blockIdx.x + 2;
	const int orig_mb_id = mb_y*cols + mb_x;

	const int left = boundaries.lefts[mb_x];
	const int top = boundaries.tops[mb_y];
	const int right = boundaries.rights[mb_x];
	const int bottom = boundaries.bottoms[mb_y];

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	uint8_t* orig_block = orig + my * w + mx;
	uint8_t* ref_search_range = ref + top*w + left;

	__shared__ int block_sads[24*32];

	int block_sad = calculate_block_sad_new<24,32>(orig_block, ref_search_range, w);

	block_sads[ref_mb_id] = block_sad;

	__syncthreads();

	min_reduce768(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, j*32+i);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu_Y_corner(uint8_t* orig, uint8_t* ref, struct corner_data* corner_datas, int cols, int w, unsigned int* index_results)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int ref_mb_id = j*blockDim.x + i;

	struct corner_data corner_data = corner_datas[blockIdx.x];

	const int mb_x = corner_data.mb_x;
	const int mb_y = corner_data.mb_y;
	const int orig_mb_id = mb_y*cols + mb_x;

	const int left = corner_data.left;
	const int top = corner_data.top;
	const int right = corner_data.right;
	const int bottom = corner_data.bottom;

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	uint8_t* orig_block = orig + my * w + mx;
	uint8_t* ref_search_range = ref + top*w + left;

	__shared__ int block_sads[16*16];

	int block_sad = calculate_block_sad_new<16,16>(orig_block, ref_search_range, w);

	block_sads[ref_mb_id] = block_sad;

	__syncthreads();

	min_reduce<16*16>(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, j*(range*2)+i);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu_Y_innercorner(uint8_t* orig, uint8_t* ref, struct corner_data* corner_datas, int cols, int w, unsigned int* index_results)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int ref_mb_id = j*blockDim.x + i;

	struct corner_data corner_data = corner_datas[blockIdx.x];

	const int mb_x = corner_data.mb_x;
	const int mb_y = corner_data.mb_y;
	const int orig_mb_id = mb_y*cols + mb_x;

	const int left = corner_data.left;
	const int top = corner_data.top;
	const int right = corner_data.right;
	const int bottom = corner_data.bottom;

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	uint8_t* orig_block = orig + my * w + mx;
	uint8_t* ref_search_range = ref + top*w + left;

	__shared__ int block_sads[24*24];

	int block_sad = calculate_block_sad_old<24,24>(orig_block, ref_search_range, w);

	block_sads[ref_mb_id] = block_sad;

	__syncthreads();

	min_reduce576(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, j*(range*2)+i);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu_Y_semicorner(uint8_t* orig, uint8_t* ref, struct corner_data* corner_datas, int cols, int w, unsigned int* index_results)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int ref_mb_id = j*blockDim.x + i;

	struct corner_data corner_data = corner_datas[blockIdx.x];

	const int mb_x = corner_data.mb_x;
	const int mb_y = corner_data.mb_y;
	const int orig_mb_id = mb_y*cols + mb_x;

	const int left = corner_data.left;
	const int top = corner_data.top;
	const int right = corner_data.right;
	const int bottom = corner_data.bottom;

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	uint8_t* orig_block = orig + my * w + mx;
	uint8_t* ref_search_range = ref + top*w + left;

	__shared__ int block_sads[16*24];

	// calculate_block_sad_new is picky on range width and height
	int block_sad = calculate_block_sad_old<24,16>(orig_block, ref_search_range, w);

	block_sads[ref_mb_id] = block_sad;

	__syncthreads();

	min_reduce384(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, j*32+i);
	}
}

template<int range>
__global__
static void me_block_8x8_gpu_Y(uint8_t* orig, uint8_t* ref, struct boundaries boundaries, int w, unsigned int* index_results)
{
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	const int ref_mb_id = j*blockDim.x + i;

	const int mb_x = blockIdx.x + 2; // Skip leftmost column
	const int mb_y = blockIdx.y + 2; // Skip the two topmost rows
	const int orig_mb_id = mb_y*(gridDim.x+4) + mb_x;

	const int left = boundaries.lefts[mb_x];
	const int top = boundaries.tops[mb_y];
	const int right = boundaries.rights[mb_x];
	const int bottom = boundaries.bottoms[mb_y];

	const int mx = mb_x * 8;
	const int my = mb_y * 8;

	uint8_t* orig_block = orig + my * w + mx;
	uint8_t* ref_search_range = ref + top*w + left;

	const int max_range_width = range * 2;
	const int max_range_height = range * 2;
	const int max_mb_count = max_range_width * max_range_height;

	__shared__ int block_sads[max_mb_count];
	int block_sad = calculate_block_sad_new<32,32>(orig_block, ref_search_range, w);
	block_sads[ref_mb_id] = block_sad;

	__syncthreads();

	min_reduce<max_mb_count>(ref_mb_id, block_sads);

	__syncthreads();

	if (block_sad == block_sads[0]) {
		atomicMin(index_results + orig_mb_id, j*32+i);
	}
}

template<int range>
__global__
static void set_motion_vectors(struct macroblock* mbs, int* lefts, int* tops, unsigned int* index_results)
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

	const int hY = cm->padh[Y_COMPONENT];

	/* Luma */
	dim3 numBlocksY(cm->mb_colsY-4, cm->mb_rowsY-4);
	dim3 threadsPerBlockY(ME_RANGE_Y*2, ME_RANGE_Y*2);

	dim3 threadsPerBlockY_corner(ME_RANGE_Y, ME_RANGE_Y);
	dim3 threadsPerBlockY_innercorner(ME_RANGE_Y+8, ME_RANGE_Y+8);
	dim3 threadsPerBlockY_semicornerH(24, 16);
	dim3 threadsPerBlockY_semicornerV(16, 24);
	dim3 threadsPerBlockY_hedge(32, 16);
	dim3 threadsPerBlockY_hinneredge(32, 24);
	dim3 threadsPerBlockY_vedge(16, 32);
	dim3 threadsPerBlockY_vinneredge(24, 32);

	dim3 blocks_horizontal(cm->mb_colsY-4, 2);
	dim3 blocks_vertical(cm->mb_rowsY-4, 2);

	cudaMemsetAsync(cm->cuda_data.sad_index_resultsY, 255, cm->mb_colsY*cm->mb_rowsY*sizeof(unsigned int), cm->cuda_data.streamY);
	me_block_8x8_gpu_Y<ME_RANGE_Y><<<numBlocksY, threadsPerBlockY, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.boundariesY, wY, cm->cuda_data.sad_index_resultsY);
	me_block_8x8_gpu_Y_corner<16><<<4, threadsPerBlockY_corner, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.corner_datasY_gpu, cm->mb_colsY, wY, cm->cuda_data.sad_index_resultsY);
	me_block_8x8_gpu_Y_semicorner<16><<<4, threadsPerBlockY_semicornerH, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.semicornerH_datasY_gpu, cm->mb_colsY, wY, cm->cuda_data.sad_index_resultsY);
	me_block_8x8_gpu_Y_semicorner<16><<<4, threadsPerBlockY_semicornerV, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.semicornerV_datasY_gpu, cm->mb_colsY, wY, cm->cuda_data.sad_index_resultsY);
	me_block_8x8_gpu_Y_outer_horizontal<16><<<blocks_horizontal, threadsPerBlockY_hedge, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.boundariesY, cm->mb_colsY, cm->mb_rowsY, wY, cm->cuda_data.sad_index_resultsY);
	me_block_8x8_gpu_Y_outer_vertical<16><<<blocks_vertical, threadsPerBlockY_vedge, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.boundariesY, cm->mb_colsY, cm->mb_rowsY, wY, cm->cuda_data.sad_index_resultsY);
	me_block_8x8_gpu_Y_innercorner<16><<<4, threadsPerBlockY_innercorner, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.innercorner_datasY_gpu, cm->mb_colsY, wY, cm->cuda_data.sad_index_resultsY);
	me_block_8x8_gpu_Y_inner_horizontal<16><<<blocks_horizontal, threadsPerBlockY_hinneredge, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.boundariesY, cm->mb_colsY, cm->mb_rowsY, wY, cm->cuda_data.sad_index_resultsY);
	me_block_8x8_gpu_Y_inner_vertical<16><<<blocks_vertical, threadsPerBlockY_vinneredge, 0, cm->cuda_data.streamY>>>(orig->Y, ref->Y, cm->cuda_data.boundariesY, cm->mb_colsY, cm->mb_rowsY, wY, cm->cuda_data.sad_index_resultsY);

	set_motion_vectors<ME_RANGE_Y><<<cm->mb_colsY, cm->mb_rowsY, 0, cm->cuda_data.streamY>>>(mbs[Y_COMPONENT], cm->cuda_data.leftsY_gpu, cm->cuda_data.topsY_gpu, cm->cuda_data.sad_index_resultsY);
	cudaMemcpyAsync(cm->curframe->mbs[Y_COMPONENT], mbs[Y_COMPONENT], cm->mb_rowsY * cm->mb_colsY * sizeof(struct macroblock), cudaMemcpyDeviceToHost, cm->cuda_data.streamY);

	/* Chroma */
	dim3 numBlocksUV(cm->mb_colsUV, cm->mb_rowsUV);
	dim3 threadsPerBlockUV(ME_RANGE_UV*2, ME_RANGE_UV*2);

	cudaMemsetAsync(cm->cuda_data.sad_index_resultsU, 255, cm->mb_colsUV*cm->mb_rowsUV*sizeof(unsigned int), cm->cuda_data.streamU);
	me_block_8x8_gpu<ME_RANGE_UV><<<numBlocksUV, threadsPerBlockUV, 0, cm->cuda_data.streamU>>>(mbs[U_COMPONENT], orig->U, ref->U, cm->cuda_data.leftsUV_gpu, cm->cuda_data.rightsUV_gpu, cm->cuda_data.topsUV_gpu, cm->cuda_data.bottomsUV_gpu, wU, cm->cuda_data.sad_index_resultsU);
	set_motion_vectors<ME_RANGE_UV><<<cm->mb_colsUV, cm->mb_rowsUV, 0, cm->cuda_data.streamU>>>(mbs[U_COMPONENT], cm->cuda_data.leftsUV_gpu, cm->cuda_data.topsUV_gpu, cm->cuda_data.sad_index_resultsU);
	cudaMemcpyAsync(cm->curframe->mbs[U_COMPONENT], mbs[U_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock), cudaMemcpyDeviceToHost, cm->cuda_data.streamU);

	cudaMemsetAsync(cm->cuda_data.sad_index_resultsV, 255, cm->mb_colsUV*cm->mb_rowsUV*sizeof(unsigned int), cm->cuda_data.streamV);
	me_block_8x8_gpu<ME_RANGE_UV><<<numBlocksUV, threadsPerBlockUV, 0, cm->cuda_data.streamV>>>(mbs[V_COMPONENT], orig->V, ref->V, cm->cuda_data.leftsUV_gpu, cm->cuda_data.rightsUV_gpu, cm->cuda_data.topsUV_gpu, cm->cuda_data.bottomsUV_gpu, wV, cm->cuda_data.sad_index_resultsV);
	set_motion_vectors<ME_RANGE_UV><<<cm->mb_colsUV, cm->mb_rowsUV, 0, cm->cuda_data.streamV>>>(mbs[V_COMPONENT], cm->cuda_data.leftsUV_gpu, cm->cuda_data.topsUV_gpu, cm->cuda_data.sad_index_resultsV);
	cudaMemcpyAsync(cm->curframe->mbs[V_COMPONENT], mbs[V_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock), cudaMemcpyDeviceToHost, cm->cuda_data.streamV);
}

/* Motion compensation for 8x8 block */
__global__
static void mc_block_8x8_gpu(struct macroblock* mbs, int w, uint8_t *predicted, uint8_t *ref)
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
