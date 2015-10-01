#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

extern "C" {
#include "dsp.h"
#include "tables.h"
}


__constant__ float dct_lookup[64] =
{
  1.0f,  0.980785f,  0.923880f,  0.831470f,  0.707107f,  0.555570f,  0.382683f,  0.195090f,
  1.0f,  0.831470f,  0.382683f, -0.195090f, -0.707107f, -0.980785f, -0.923880f, -0.555570f,
  1.0f,  0.555570f, -0.382683f, -0.980785f, -0.707107f,  0.195090f,  0.923880f,  0.831470f,
  1.0f,  0.195090f, -0.923880f, -0.555570f,  0.707107f,  0.831470f, -0.382683f, -0.980785f,
  1.0f, -0.195090f, -0.923880f,  0.555570f,  0.707107f, -0.831470f, -0.382683f,  0.980785f,
  1.0f, -0.555570f, -0.382683f,  0.980785f, -0.707107f, -0.195090f,  0.923880f, -0.831470f,
  1.0f, -0.831470f,  0.382683f,  0.195090f, -0.707107f,  0.980785f, -0.923880f,  0.555570f,
  1.0f, -0.980785f,  0.923880f, -0.831470f,  0.707107f, -0.555570f,  0.382683f, -0.195090f
};


/* Array containing the indexes resulting from calculating
 * (zigzag_V[zigzag]*8) + zigzag_U[zigzag] for zigzag = 0, 1, ..., 63
 */
__constant__ uint8_t UV_indexes[64] =
{
	 0,  1,  8, 16,  9,  2,  3, 10,
	17, 24, 32, 25, 18, 11,  4,  5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13,  6,  7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63,
};

__shared__ float dct_macro_block[64];
__shared__ float dct_macro_block2[64];
__shared__ float idct_macro_block[64];
__shared__ float idct_macro_block2[64];



float *cuda_mb, *cuda_mb2;
int16_t *cuda_in_data, *cuda_out_data;
uint8_t *cuda_quant_tbl;

//__device__ float dct_lookup[64];


static void transpose_block(float *in_data, float *out_data)
{
	int i, j;
	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			out_data[i * 8 + j] = in_data[j * 8 + i];
		}
	}
}

static void dct_1d(float *in_data, float *out_data)
{
	int i, j;
	for (i = 0; i < 8; ++i)
	{
		float dct = 0;

		for (j = 0; j < 8; ++j)
		{
			dct += in_data[j] * dctlookup[j*8+i];
		}

		out_data[i] = dct;
	}
}

static void idct_1d(float *in_data, float *out_data)
{
	int i, j;
	for (i = 0; i < 8; ++i)
	{
		float idct = 0;

		for (j = 0; j < 8; ++j)
		{
			idct += in_data[j] * dctlookup[i*8+j];
		}

		out_data[i] = idct;
	}
}

__host__ void cuda_init() {
	
	cudaMalloc(&cuda_mb, 64*sizeof(float));
	cudaMalloc(&cuda_mb2, 64*sizeof(float));
	
	cudaMalloc(&cuda_in_data, 64*sizeof(int16_t));
	cudaMalloc(&cuda_out_data, 64*sizeof(int16_t));
	
	cudaMalloc(&cuda_quant_tbl, 64*sizeof(uint8_t));
	
	//cudaMemcpy((float*)dct_lookup, (float*)dctlookup, 64*sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void cuda_cleanup() {
	cudaFree(cuda_mb);
	cudaFree(cuda_mb2);
	
	cudaFree(cuda_in_data);
	cudaFree(cuda_out_data);
	
	cudaFree(cuda_quant_tbl);
}


__global__ void gpu_dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl, float *mb, float *mb2)
{
	int i = threadIdx.y;
	int j = threadIdx.x;
	
	
	// Copy pixel to shared memory
	//dct_macro_block2[i*8+j] = in_data[i*8+j];
	dct_macro_block2[i*8+j] = mb2[i*8+j];
	dct_macro_block[i*8+j] = mb[i*8+j];
	__syncthreads();
	
	/*
	// First dct_1d - mb = mb2
	float dct = 0;
	int k;	
	for (k = 0; k < 8; ++k) {
		dct += dct_macro_block2[i*8+k] * dct_lookup[k*8+j];
	}
	dct_macro_block[i*8+j] = dct;
	__syncthreads();
	
	
	// First transpose - mb2 = mb
	dct_macro_block2[i * 8 + j] = dct_macro_block[j * 8 + i];
	__syncthreads();
	
	
	// Second dct_1d - mb = mb2
	dct = 0;
	for (k = 0; k < 8; ++k) {
		dct += dct_macro_block2[i*8+k] * dct_lookup[k*8+j];
	}
	dct_macro_block[i*8+j] = dct;
	__syncthreads();
	
	
	// Second transpose - mb2 = mb
	dct_macro_block2[i * 8 + j] = dct_macro_block[j * 8 + i];
	__syncthreads();
	*/
	
	// Scale
	dct_macro_block[i*8+j] = dct_macro_block2[i*8+j];	
	if(j == 0) {
		dct_macro_block[i*8+j] *= ISQRT2;
	}
	if(i == 0) {
		dct_macro_block[i*8+j] *= ISQRT2;
	}
	__syncthreads();
	
	
	// Quantize	
	float dct = dct_macro_block[UV_indexes[i*8+j]];
	dct_macro_block2[i*8+j] = (float) round((dct/4.0) / quant_tbl[i*8+j]);
	__syncthreads();
	
	
	// Set value in cuda_out_data
	out_data[i*8+j] = dct_macro_block2[i*8+j];
	__syncthreads();
}


__host__ void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
	float mb[64] __attribute__((aligned(16)));
	float mb2[64] __attribute__((aligned(16)));
	
	int i;
	for (i = 0; i < 64; ++i) {
		mb2[i] = in_data[i];
	}
	
	/* Two 1D DCT operations with transpose */
	int v;
	for (v = 0; v < 8; ++v)
	{
		dct_1d(mb2 + v * 8, mb + v * 8);
	}
	transpose_block(mb, mb2);
	for (v = 0; v < 8; ++v)
	{
		dct_1d(mb2 + v * 8, mb + v * 8);
	}
	transpose_block(mb, mb2);
	
	cudaMemcpy(cuda_mb, (float*)mb , 64*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(cuda_mb2, (float*)mb2, 64*sizeof(float), cudaMemcpyHostToDevice);
	
	
	
	// Copy in_data and quant table to gpu memory
	cudaMemcpy(cuda_in_data, in_data , 64*sizeof(int16_t), cudaMemcpyHostToDevice); 
	cudaMemcpy(cuda_quant_tbl, quant_tbl, 64*sizeof(uint8_t), cudaMemcpyHostToDevice);
	
	// Set number of blocks and number of threads per block
	int numBlocks = 1;
	dim3 threadsPerBlock(8, 8);
	
	// Call gpu kernel - one thread works on one pixel
	gpu_dct_quant_block_8x8<<<numBlocks, threadsPerBlock>>>(cuda_in_data, cuda_out_data, cuda_quant_tbl, cuda_mb, cuda_mb2);
	
	// Copy out data from gpu memory to host mempory
	cudaMemcpy(out_data, cuda_out_data, 64*sizeof(int16_t), cudaMemcpyDeviceToHost);
}



__global__ void gpu_dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl, float *mb, float *mb2)
{
	int i = threadIdx.y;
	int j = threadIdx.x;
	
	// Copy to shared memory
	idct_macro_block[i*8+j] = in_data[i*8+j];
	__syncthreads();
	
	
	// Dequantize
	float dct = idct_macro_block[i*8+j];
	idct_macro_block2[UV_indexes[i*8+j]] = (float) round((dct*quant_tbl[i*8+j]) / 4.0);
	__syncthreads();
	
	
	// Scale
	idct_macro_block[i*8+j] = idct_macro_block2[i*8+j];
	if(j == 0) {
		idct_macro_block[i*8+j] *= ISQRT2;
	}
	if(i == 0) {
		idct_macro_block[i*8+j] *= ISQRT2;
	}
	__syncthreads();
		
	
	// First idct - mb2 = mb
	float idct = 0;
	int k;
	for (k = 0; k < 8; ++k) {
		idct += idct_macro_block[i*8+k] * dct_lookup[j*8+k];
	}
	idct_macro_block2[i*8+j] = idct;
	__syncthreads();
	
	/*
	// First transpose - mb = mb2
	idct_macro_block[i*8+j] = idct_macro_block2[j*8+i];
	__syncthreads();
	
	// Second idct - mb2 = mb
	idct = 0;
	for (k = 0; k < 8; ++k) {
		idct += idct_macro_block[i*8+k] * dct_lookup[j*8+k];
	}
	idct_macro_block2[i*8+j] = idct;
	__syncthreads();
	
	
	// Second transpose - mb = mb2
	idct_macro_block[i * 8 + j] = idct_macro_block2[j * 8 + i];
	__syncthreads();
	
	*/
	// Copy to out_data
	//out_data[i*8+j] = idct_macro_block[i*8+j];
	//in_data[UV_indexes[i*8+j]] = idct_macro_block2[UV_indexes[i*8+j]];
	mb[i*8+j] = idct_macro_block[i*8+j];
	mb2[UV_indexes[i*8+j]] = idct_macro_block2[UV_indexes[i*8+j]];
	mb2[i*8+j] = idct_macro_block2[i*8+j];
	__syncthreads();
	
}



__host__ void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
	// Copy in_data and quant_tbl to gpu memory
	cudaMemcpy(cuda_in_data, in_data, 64*sizeof(int16_t), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_quant_tbl, quant_tbl, 64*sizeof(uint8_t), cudaMemcpyHostToDevice);
	
	// Set number of blocks and number of threads per block
	int numBlocks = 1;
	dim3 threadsPerBlock(8, 8);
	
	// Call gpu kernel - one thread works on one pixel
	gpu_dequant_idct_block_8x8<<<numBlocks, threadsPerBlock>>>(cuda_in_data, cuda_out_data, cuda_quant_tbl, cuda_mb, cuda_mb2);
	
	// Copy out_data from gpu memory to host memory
	cudaMemcpy(out_data, cuda_out_data, 64*sizeof(int16_t), cudaMemcpyDeviceToHost);	
	
	float mb[64] __attribute__((aligned(16)));
	float mb2[64] __attribute__((aligned(16)));
	
	cudaMemcpy(mb, cuda_mb, 64*sizeof(float), cudaMemcpyDeviceToHost);	
	cudaMemcpy(mb2, cuda_mb2, 64*sizeof(float), cudaMemcpyDeviceToHost);	
	
	/* Two 1D inverse DCT operations with transpose */
	int v;
	/*
	for (v = 0; v < 8; ++v)
	{
		idct_1d(mb + v * 8, mb2 + v * 8);
	}*/
	transpose_block(mb2, mb);
	for (v = 0; v < 8; ++v)
	{
		idct_1d(mb + v * 8, mb2 + v * 8);
	}
	transpose_block(mb2, mb);
	int i;
	for (i = 0; i < 64; ++i)
	{
		out_data[i] = mb[i];
	}
}




void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
	int u, v;

	*result = 0;

	for (v = 0; v < 8; ++v)
	{
		for (u = 0; u < 8; ++u)
		{
			*result += abs(block2[v * stride + u] - block1[v * stride + u]);
		}
	}
}
