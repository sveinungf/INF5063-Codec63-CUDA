#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63_cuda.h"
#include "dsp.h"

namespace gpu = c63::gpu;
using namespace gpu;

static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

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
static void dct_quantize_gpu(const uint8_t* __restrict__ in_data,
		const uint8_t* __restrict__ prediction, int w, int16_t* __restrict__ out_data,
		int quantization)
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
static void dequantize_idct_gpu(const int16_t* __restrict__ in_data,
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

template<int component>
void gpu::dct_quantize(struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
	const int w = cm->padw[component];
	const int h = cm->padh[component];
	const cudaStream_t stream = c63_cuda.stream[component];

	uint8_t* orig;
	uint8_t* predicted;
	int16_t* residuals;
	int16_t* residuals_host;

	struct frame* f = cm->curframe;

	switch (component)
	{
		case Y_COMPONENT:
			orig = (uint8_t*) f->orig_gpu->Y;
			predicted = f->predicted_gpu->Y;
			residuals = f->residuals_gpu->Ydct;
			residuals_host = f->residuals->Ydct;
			break;
		case U_COMPONENT:
			orig = (uint8_t*) f->orig_gpu->U;
			predicted = f->predicted_gpu->U;
			residuals = f->residuals_gpu->Udct;
			residuals_host = f->residuals->Udct;
			break;
		case V_COMPONENT:
			orig = (uint8_t*) f->orig_gpu->V;
			predicted = f->predicted_gpu->V;
			residuals = f->residuals_gpu->Vdct;
			residuals_host = f->residuals->Vdct;
			break;
	}

	const dim3 threadsPerBlock(8, 8);
	const dim3 numBlocks(w / threadsPerBlock.x, h / threadsPerBlock.y);

	dct_quantize_gpu<<<numBlocks, threadsPerBlock, 0, stream>>>(orig, predicted, w, residuals,
			component);

	cudaEvent_t dctquant_done = c63_cuda.dctquant_done[component];
	cudaEventRecord(dctquant_done, stream);

	cudaStream_t memcpy_stream = c63_cuda.memcpy_stream[component];
	cudaStreamWaitEvent(memcpy_stream, dctquant_done, 0);
	cudaMemcpyAsync(residuals_host, residuals, w * h * sizeof(int16_t), cudaMemcpyDeviceToHost,
			memcpy_stream);
}

template<int component>
void gpu::dequantize_idct(struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
	const int w = cm->padw[component];
	const int h = cm->padh[component];
	const cudaStream_t stream = c63_cuda.stream[component];

	uint8_t* predicted;
	uint8_t* recons;
	int16_t* residuals;

	struct frame* f = cm->curframe;

	switch (component)
	{
		case Y_COMPONENT:
			predicted = f->predicted_gpu->Y;
			recons = f->recons_gpu->Y;
			residuals = f->residuals_gpu->Ydct;
			break;
		case U_COMPONENT:
			predicted = f->predicted_gpu->U;
			recons = f->recons_gpu->U;
			residuals = f->residuals_gpu->Udct;
			break;
		case V_COMPONENT:
			predicted = f->predicted_gpu->V;
			recons = f->recons_gpu->V;
			residuals = f->residuals_gpu->Vdct;
			break;
	}

	const dim3 threadsPerBlock(8, 8);
	const dim3 numBlocks(w / threadsPerBlock.x, h / threadsPerBlock.y);

	dequantize_idct_gpu<<<numBlocks, threadsPerBlock, 0, stream>>>(residuals, predicted, w, recons,
			component);
}

template void gpu::dct_quantize<Y>(struct c63_common* cm, const struct c63_cuda& c63_cuda);
template void gpu::dct_quantize<U>(struct c63_common* cm, const struct c63_cuda& c63_cuda);
template void gpu::dct_quantize<V>(struct c63_common* cm, const struct c63_cuda& c63_cuda);

template void gpu::dequantize_idct<Y>(struct c63_common* cm, const struct c63_cuda& c63_cuda);
template void gpu::dequantize_idct<U>(struct c63_common* cm, const struct c63_cuda& c63_cuda);
template void gpu::dequantize_idct<V>(struct c63_common* cm, const struct c63_cuda& c63_cuda);
