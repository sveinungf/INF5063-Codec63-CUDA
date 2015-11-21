#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"

namespace gpu = c63::gpu;
using namespace gpu;

__device__
static void dct_quant_block_8x8(float* in_data, float *out_data, int16_t __restrict__ *global_out,
		const uint8_t* __restrict__ quant_tbl, int i, const int j)
{
	// First dct_1d - mb = mb2 - and transpose
	float dct = 0;
	int k;
#pragma unroll
	for (k = 0; k < 8; ++k)
	{
		dct += in_data[j * 8 + k] * dct_lookup_trans[i * 8 + k];
	}
	out_data[i * 8 + j] = dct;
	__syncthreads();

	// Second dct_1d - mb = mb2 - and transpose
	dct = 0;
#pragma unroll
	for (k = 0; k < 8; ++k)
	{
		dct += out_data[j * 8 + k] * dct_lookup_trans[i * 8 + k];
	}

	// Scale
	if (i == 0)
	{
		dct *= ISQRT2;
	}
	if (j == 0)
	{
		dct *= ISQRT2;
	}
	in_data[i * 8 + j] = dct;
	__syncthreads();

	// Quantize and set value in out_data
	dct = in_data[UV_indexes[i * 8 + j]];
	global_out[i * 8 + j] = round((dct / 4.0f) / quant_tbl[i * 8 + j]);
}

__global__
void gpu::dct_quantize(const uint8_t* __restrict__ in_data, const uint8_t* __restrict__ prediction,
		int w, int16_t* __restrict__ out_data, int quantization)
{
	const int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

	const int i = threadIdx.y;
	const int j = threadIdx.x;

	const int offset = blockIdx.x * blockDim.x + blockIdx.y * w * blockDim.y + i * w + j;

	__shared__ float dct_in[65];
	__shared__ float dct_out[65];

	dct_in[i * 8 + j] = ((float) in_data[offset] - prediction[block_offset + i * 8 + j]);
	__syncthreads();
	dct_quant_block_8x8(dct_in, dct_out, out_data + block_offset, quant_table + quantization * 64,
			i, j);
}

__device__
static void dequant_idct_block_8x8(float *in_data, float *out_data,
		const uint8_t* __restrict__ quant_tbl, int i, int j)
{
	// Dequantize
	float dct = in_data[i * 8 + j];
	out_data[UV_indexes[i * 8 + j]] = (float) round((dct * quant_tbl[i * 8 + j]) / 4.0f);
	__syncthreads();

	// Scale
	if (i == 0)
	{
		out_data[i * 8 + j] *= ISQRT2;
	}
	if (j == 0)
	{
		out_data[i * 8 + j] *= ISQRT2;
	}
	in_data[i * 8 + j] = out_data[i * 8 + j];
	__syncthreads();

	// First idct - mb2 = mb - and transpose
	float idct = 0;
	int k;
#pragma unroll
	for (k = 0; k < 8; ++k)
	{
		idct += in_data[j * 8 + k] * dct_lookup[i * 8 + k];
	}

	out_data[i * 8 + j] = idct;
	__syncthreads();

	// Second idct - mb2 = mb - and transpose
	idct = 0;
#pragma unroll
	for (k = 0; k < 8; ++k)
	{
		idct += out_data[j * 8 + k] * dct_lookup[i * 8 + k];
	}
	in_data[i * 8 + j] = idct;
}

__global__
void gpu::dequantize_idct(const int16_t* __restrict__ in_data,
		const uint8_t* __restrict__ prediction, int w, uint8_t* __restrict__ out_data,
		int quantization)
{
	const int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

	const int i = threadIdx.y;
	const int j = threadIdx.x;

	__shared__ float idct_in[65];
	__shared__ float idct_out[65];

	idct_in[i * 8 + j] = (float) in_data[block_offset + i * 8 + j];
	__syncthreads();

	dequant_idct_block_8x8(idct_in, idct_out, quant_table + quantization * 64, i, j);

	const int offset = blockIdx.x * 8 + blockIdx.y * w * 8 + i * w + j;

	int16_t tmp = (int16_t) idct_in[i * 8 + j] + (int16_t) prediction[block_offset + i * 8 + j];

	if (tmp < 0)
	{
		tmp = 0;
	}
	else if (tmp > 255)
	{
		tmp = 255;
	}

	out_data[offset] = tmp;
}
