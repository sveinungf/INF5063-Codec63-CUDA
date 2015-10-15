#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "dsp.h"


__device__
static void dct_quant_block_8x8(float* in_data, float *out_data, int16_t __restrict__ *global_out, const uint8_t* __restrict__ quant_tbl, int i, const int j)
{
	// First dct_1d - mb = mb2 - and transpose
	float dct = 0;
	int k;
	#pragma unroll
	for (k = 0; k < 8; ++k) {
		dct += in_data[j*8+k] * dct_lookup_trans[i*8+k];
	}
	out_data[i*8+j] = dct;
	__syncthreads();

	// Second dct_1d - mb = mb2 - and transpose
	dct = 0;
	#pragma unroll
	for (k = 0; k < 8; ++k) {
		dct += out_data[j*8+k] * dct_lookup_trans[i*8+k];
	}

	// Scale
	if(i == 0) {
		dct *= ISQRT2;
	}
	if(j == 0) {
		dct *= ISQRT2;
	}
	in_data[i*8+j] = dct;
	__syncthreads();

	// Quantize and set value in out_data
	dct = in_data[UV_indexes[i*8+j]];
	global_out[i*8+j] = round((dct/4.0f) / quant_tbl[i*8+j]);
}

__global__
void dct_quantize(const uint8_t* __restrict__ in_data, const uint8_t* __restrict__ prediction, int w, int16_t* __restrict__ out_data, int quantization)
{
	const int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

	const int i = threadIdx.y;
	const int j = threadIdx.x;

	const int offset = blockIdx.x * blockDim.x  + blockIdx.y * w*blockDim.y + i*w+j;

	__shared__ float dct_in[65];
	__shared__ float dct_out[65];

	dct_in[i*8+j] = ((float) in_data[offset] - prediction[block_offset + i*8+j]);
	__syncthreads();
	dct_quant_block_8x8(dct_in, dct_out, out_data + block_offset, quant_table + quantization*64, i, j);
}


__device__
static void dequant_idct_block_8x8(float *in_data, float *out_data, const uint8_t* __restrict__ quant_tbl, int i, int j)
{
	// Dequantize
	float dct = in_data[i*8+j];
	out_data[UV_indexes[i*8+j]] = (float) round((dct*quant_tbl[i*8+j]) / 4.0f);
	__syncthreads();

	// Scale
	if(i == 0) {
		out_data[i*8+j] *= ISQRT2;
	}
	if(j == 0) {
		out_data[i*8+j] *= ISQRT2;
	}
	in_data[i*8+j] = out_data[i*8+j];
	__syncthreads();

	// First idct - mb2 = mb - and transpose
	float idct = 0;
	int k;
	#pragma unroll
	for (k = 0; k < 8; ++k) {
		idct += in_data[j*8+k] * dct_lookup[i*8+k];
	}

	out_data[i*8+j] = idct;
	__syncthreads();

	// Second idct - mb2 = mb - and transpose
	idct = 0;
	#pragma unroll
	for (k = 0; k < 8; ++k) {
		idct += out_data[j*8+k] * dct_lookup[i*8+k];
	}
	in_data[i*8+j] = idct;
}

__global__
void dequantize_idct(const int16_t* __restrict__ in_data, const uint8_t* __restrict__ prediction, int w, uint8_t* __restrict__ out_data, int quantization)
{
	const int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

 	const int i = threadIdx.y;
 	const int j = threadIdx.x;

 	__shared__ float idct_in[65];
 	__shared__ float idct_out[65];

 	idct_in[i*8+j] = (float) in_data[block_offset + i*8+j];
 	__syncthreads();

	dequant_idct_block_8x8(idct_in, idct_out, quant_table + quantization*64, i, j);

	const int offset = blockIdx.x * 8 + blockIdx.y * w*8 + i*w+j;

	int16_t tmp = (int16_t) idct_in[i*8+j] + (int16_t) prediction[block_offset+i*8+j];

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


static void init_frame_gpu(struct c63_common* cm, struct frame* f)
{
	f->recons_gpu = create_image_gpu(cm);
	f->predicted_gpu = create_image_gpu(cm);

	f->residuals_gpu = (dct_t*) malloc(sizeof(dct_t));
	cudaMalloc((void**) &f->residuals_gpu->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
	cudaMalloc((void**) &f->residuals_gpu->Udct, cm->upw * cm->uph * sizeof(int16_t));
	cudaMalloc((void**) &f->residuals_gpu->Vdct, cm->vpw * cm->vph * sizeof(int16_t));

	cudaMalloc((void**) &f->mbs_gpu[Y_COMPONENT], cm->mb_rowsY * cm->mb_colsY *
			sizeof(struct macroblock));
	cudaMalloc((void**) &f->mbs_gpu[U_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV *
			sizeof(struct macroblock));
	cudaMalloc((void**) &f->mbs_gpu[V_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV *
			sizeof(struct macroblock));
}

static void deinit_frame_gpu(struct frame* f)
{
	destroy_image_gpu(f->recons_gpu);
	destroy_image_gpu(f->predicted_gpu);

	cudaFree(f->residuals_gpu->Ydct);
	cudaFree(f->residuals_gpu->Udct);
	cudaFree(f->residuals_gpu->Vdct);
	free(f->residuals_gpu);

	cudaFree(f->mbs_gpu[Y_COMPONENT]);
	cudaFree(f->mbs_gpu[U_COMPONENT]);
	cudaFree(f->mbs_gpu[V_COMPONENT]);
}

struct macroblock *create_mb(struct macroblock *mb, size_t size) {
	cudaMallocHost((void**)&mb, size);
	cudaMemset(mb, 0, size);

	return mb;
}

struct frame* create_frame(struct c63_common *cm)
{
	struct frame *f = (frame*) malloc(sizeof(struct frame));

	f->residuals = (dct_t*) malloc(sizeof(dct_t));
	cudaMallocHost((void**)&f->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
	cudaMallocHost((void**)&f->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
	cudaMallocHost((void**)&f->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));

	//f->mbs[Y_COMPONENT] = (struct macroblock*) calloc(cm->mb_rowsY * cm->mb_colsY, sizeof(struct macroblock));
	//f->mbs[U_COMPONENT] = (struct macroblock*) calloc(cm->mb_rowsUV * cm->mb_colsUV, sizeof(struct macroblock));
	//f->mbs[V_COMPONENT] = (struct macroblock*) calloc(cm->mb_rowsUV * cm->mb_colsUV, sizeof(struct macroblock));
	f->mbs[Y_COMPONENT] = create_mb(f->mbs[Y_COMPONENT], cm->mb_rowsY * cm->mb_colsY * sizeof(struct macroblock));
	f->mbs[U_COMPONENT] = create_mb(f->mbs[U_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock));
	f->mbs[V_COMPONENT] = create_mb(f->mbs[V_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV * sizeof(struct macroblock));

	init_frame_gpu(cm, f);

	return f;
}

void destroy_frame(struct frame *f)
{
	deinit_frame_gpu(f);

	cudaFreeHost(f->residuals->Ydct);
	cudaFreeHost(f->residuals->Udct);
	cudaFreeHost(f->residuals->Vdct);
	free(f->residuals);

	cudaFreeHost(f->mbs[Y_COMPONENT]);
	cudaFreeHost(f->mbs[U_COMPONENT]);
	cudaFreeHost(f->mbs[V_COMPONENT]);

	free(f);
}

yuv_t* create_image(struct c63_common *cm)
{
	yuv_t* image = (yuv_t*) malloc(sizeof(yuv_t));
	cudaHostAlloc((void**)&image->Y, cm->ypw * cm->yph * sizeof(uint8_t), cudaHostAllocWriteCombined);
	cudaHostAlloc((void**)&image->U, cm->upw * cm->uph * sizeof(uint8_t), cudaHostAllocWriteCombined);
	cudaHostAlloc((void**)&image->V, cm->vpw * cm->vph * sizeof(uint8_t), cudaHostAllocWriteCombined);
	//image->Y = (uint8_t*) malloc(cm->ypw * cm->yph * sizeof(uint8_t));
	//image->U = (uint8_t*) malloc(cm->upw * cm->uph * sizeof(uint8_t));
	//image->V = (uint8_t*) malloc(cm->vpw * cm->vph * sizeof(uint8_t));

	return image;
}

void destroy_image(yuv_t *image)
{
	cudaFreeHost(image->Y);
	cudaFreeHost(image->U);
	cudaFreeHost(image->V);
	free(image);
}

yuv_t* create_image_gpu(struct c63_common *cm)
{
	yuv_t* image = (yuv_t*) malloc(sizeof(yuv_t));
	cudaMalloc((void**) &image->Y, cm->ypw * cm->yph * sizeof(uint8_t));
	cudaMalloc((void**) &image->U, cm->upw * cm->uph * sizeof(uint8_t));
	cudaMalloc((void**) &image->V, cm->vpw * cm->vph * sizeof(uint8_t));

	return image;
}

void destroy_image_gpu(yuv_t* image)
{
	cudaFree(image->Y);
	cudaFree(image->U);
	cudaFree(image->V);
	free(image);
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
	fwrite(image->Y, 1, w * h, fp);
	fwrite(image->U, 1, w * h / 4, fp);
	fwrite(image->V, 1, w * h / 4, fp);
}
