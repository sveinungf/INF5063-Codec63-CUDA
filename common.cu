extern "C" {
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

#include "common.h"
#include "dsp.h"


__shared__ float dct_in[64];
__shared__ float dct_out[64];
__shared__ float idct_in[64];
__shared__ float idct_out[64];


__global__
void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, uint8_t *out_data, int quantization)
{
	int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

 	int i = threadIdx.y;
 	int j = threadIdx.x;

 	idct_in[i*8+j] = (float) in_data[block_offset + i*8+j];
 	__syncthreads();

	dequant_idct_block_8x8(idct_in, idct_out, quantization, i, j);

	int offset = blockIdx.x * 8 + blockIdx.y * w*8 + i * w + j;

	int16_t tmp = (int16_t) idct_out[i*8+j] + (int16_t) (prediction[offset]);
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

__host__
void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		uint8_t *gpu_out_data, int quantization)
{
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	dequantize_idct_row<<<numBlocks, threadsPerBlock>>>(in_data, prediction, width, gpu_out_data, quantization);
}


__global__
void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int16_t *out_data,
		int quantization)
{
	int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

	int i = threadIdx.y;
	int j = threadIdx.x;

	int offset = blockIdx.x * 8 + blockIdx.y * w*8 + i*w+j;

	dct_in[i*8+j] = ((float) in_data[offset] - prediction[offset]);
	__syncthreads();
	dct_quant_block_8x8(dct_in, dct_out, quantization, i, j);

	out_data[block_offset + i*8+j] = dct_in[i*8+j];
}

__host__
void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		int16_t *gpu_out_data, int16_t *out_data, int quantization)
{
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	dct_quantize_row<<<numBlocks, threadsPerBlock>>>(in_data, prediction, width,
				gpu_out_data, quantization);
	cudaMemcpy(out_data, gpu_out_data, width*height*sizeof(int16_t), cudaMemcpyDeviceToHost);
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

struct frame* create_frame(struct c63_common *cm)
{
	struct frame *f = (frame*) malloc(sizeof(struct frame));

	f->predicted = create_image(cm);

	f->residuals = (dct_t*) malloc(sizeof(dct_t));
	f->residuals->Ydct = (int16_t*) malloc(cm->ypw * cm->yph * sizeof(int16_t));
	f->residuals->Udct = (int16_t*) malloc(cm->upw * cm->uph * sizeof(int16_t));
	f->residuals->Vdct = (int16_t*) malloc(cm->vpw * cm->vph * sizeof(int16_t));

	f->mbs[Y_COMPONENT] = (struct macroblock*) calloc(cm->mb_rowsY * cm->mb_colsY, sizeof(struct macroblock));
	f->mbs[U_COMPONENT] = (struct macroblock*) calloc(cm->mb_rowsUV * cm->mb_colsUV, sizeof(struct macroblock));
	f->mbs[V_COMPONENT] = (struct macroblock*) calloc(cm->mb_rowsUV * cm->mb_colsUV, sizeof(struct macroblock));

	init_frame_gpu(cm, f);

	return f;
}

void destroy_frame(struct frame *f)
{
	deinit_frame_gpu(f);

	destroy_image(f->predicted);

	free(f->residuals->Ydct);
	free(f->residuals->Udct);
	free(f->residuals->Vdct);
	free(f->residuals);

	free(f->mbs[Y_COMPONENT]);
	free(f->mbs[U_COMPONENT]);
	free(f->mbs[V_COMPONENT]);

	free(f);
}

yuv_t* create_image(struct c63_common *cm)
{
	yuv_t* image = (yuv_t*) malloc(sizeof(yuv_t));
	image->Y = (uint8_t*) malloc(cm->ypw * cm->yph * sizeof(uint8_t));
	image->U = (uint8_t*) malloc(cm->upw * cm->uph * sizeof(uint8_t));
	image->V = (uint8_t*) malloc(cm->vpw * cm->vph * sizeof(uint8_t));

	return image;
}

void destroy_image(yuv_t *image)
{
	free(image->Y);
	free(image->U);
	free(image->V);
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
