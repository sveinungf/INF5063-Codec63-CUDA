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
#include <cuda_runtime.h>
#include <cuda.h>

int16_t *gpu_frame1_16;
int16_t *gpu_frame2_16;
int16_t *gpu_out_16;

uint8_t *gpu_out_8;
uint8_t *gpu_in_8;
uint8_t *gpu_prediction_8;

void cuda_init(int width, int height){
	cudaMalloc(&gpu_frame1_16, width*height*sizeof(int16_t));
	cudaMalloc(&gpu_frame2_16, width*height*sizeof(int16_t));
	cudaMalloc(&gpu_out_16, width*height*sizeof(int16_t));

	cudaMalloc(&gpu_out_8, width*height*sizeof(uint8_t));
	cudaMalloc(&gpu_in_8, width*height*sizeof(uint8_t));
	cudaMalloc(&gpu_prediction_8, width*height*sizeof(uint8_t));
}

void cuda_cleanup() {
	cudaFree(gpu_frame1_16);
	cudaFree(gpu_frame2_16);
	cudaFree(gpu_out_16);

	cudaFree(gpu_in_8);
	cudaFree(gpu_out_8);
	cudaFree(gpu_prediction_8);
}

__global__ void dequantize_add_pred(int16_t *in_data, uint8_t *prediction, int w, int16_t *out_data) {

	int block_offset = blockIdx.x * 64;
	int x_offset = blockIdx.x * 8;

	int i = threadIdx.y;
	int j = threadIdx.x;

	/* Add prediction block. Note: DCT is not precise -
	Clamp to legal values */
	out_data[i * w + j + x_offset] = in_data[block_offset + i * 8 + j] + (int16_t) prediction[i * w + j + x_offset];
}

__global__ void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h, int y,
		int16_t *out_data, int quantization)
{
	int block_offset = blockIdx.x * 64;
 	int x_offset = blockIdx.x * 8;

 	int i = threadIdx.y;
 	int j = threadIdx.x;

	dequant_idct_block_8x8(in_data, out_data, quantization, block_offset, i, j);
	//__syncthreads();
	//out_data[i * w + j + x_offset] = temp[block_offset + i * 8 + j] + (int16_t) prediction[i * w + j + x_offset];
	//__syncthreads();
	//dequantize_end<<<numBlocks, threadsPerBlock>>>(gpu_frame2_16 + y * w, prediction, w, out_data);
	/*
	int16_t block[w*8];
	cudaMemcpy(block, gpu_frame2_16 + y * w, w*8*sizeof(int16_t), cudaMemcpyDeviceToHost);

	// Perform the dequantization and iDCT
	int x;
	for (x = 0; x < w; x += 8)
	{
		int i, j;

		for (i = 0; i < 8; ++i)
		{
			for (j = 0; j < 8; ++j)
			{
				// Add prediction block. Note: DCT is not precise -
				// Clamp to legal values
				int16_t tmp = block[i * 8 + j + x * 8] + (int16_t) prediction[i * w + j + x];

				if (tmp < 0)
				{
					tmp = 0;
				}
				else if (tmp > 255)
				{
					tmp = 255;
				}

				out_data[i * w + j + x] = tmp;
			}
		}
	}*/
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		uint8_t *out_data, int quantization)
{
	int numBlocks = width/8;
	dim3 threadsPerBlock(8, 8);

	unsigned int y;
	for (y = 0; y < height; y += 8)
	{
		dequantize_idct_row<<<numBlocks, threadsPerBlock>>>(in_data + y * width, prediction + y * width, width, height, y,
				gpu_frame1_16 + y * width, quantization);
		dequantize_add_pred<<<numBlocks, threadsPerBlock>>>(gpu_frame1_16 + y * width, prediction + y * width, width, gpu_out_16 + y * width);
	}

	int16_t temp[width*height];
	cudaMemcpy(temp, gpu_out_16, width*height*sizeof(int16_t), cudaMemcpyDeviceToHost);

	for(y = 0; y < width*height; ++y) {

		out_data[y] = temp[y];

		if (temp[y] < 0)
		{
			out_data[y] = 0;
		}
		else if (temp[y] > 255)
		{
			out_data[y] = 255;
		}
	}
}

__global__ void dct_quantize_row(uint8_t *in_data, int16_t *temp, uint8_t *prediction, int w, int h, int16_t *out_data,
		int quantization)
{
 	int block_offset = blockIdx.x * 64;
 	int x_offset = blockIdx.x * 8;

 	int i = threadIdx.y;
 	int j = threadIdx.x;

	temp[block_offset + i * 8 + j] = ((int16_t) in_data[i * w + j + x_offset] - prediction[i * w + j + x_offset]);

 	dct_quant_block_8x8(temp, out_data, quantization, block_offset, i, j);
}


void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		int16_t *out_data, int quantization)
{
	int numBlocks = width/8;
	dim3 threadsPerBlock(8, 8);

	unsigned int y;
	for (y = 0; y < height; y += 8)
	{
		dct_quantize_row<<<numBlocks, threadsPerBlock>>>(in_data + y * width, gpu_frame1_16 + y * width, prediction + y * width, width, height,
				gpu_out_16 + y * width, quantization);
	}
	cudaMemcpy(out_data, gpu_out_16, width*height*sizeof(int16_t), cudaMemcpyDeviceToHost);
}

void destroy_frame(struct frame *f)
{
	/* First frame doesn't have a reconstructed frame to destroy */
	if (!f)
	{
		return;
	}

	free(f->recons->Y);
	free(f->recons->U);
	free(f->recons->V);
	free(f->recons);

	free(f->residuals->Ydct);
	free(f->residuals->Udct);
	free(f->residuals->Vdct);
	free(f->residuals);

	free(f->predicted->Y);
	free(f->predicted->U);
	free(f->predicted->V);
	free(f->predicted);

	free(f->mbs[Y_COMPONENT]);
	free(f->mbs[U_COMPONENT]);
	free(f->mbs[V_COMPONENT]);

	free(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
	struct frame *f = (frame*) malloc(sizeof(struct frame));

	f->orig = image;

	f->recons = (yuv_t*) malloc(sizeof(yuv_t));
	f->recons->Y = (uint8_t*) malloc(cm->ypw * cm->yph);
	f->recons->U = (uint8_t*) malloc(cm->upw * cm->uph);
	f->recons->V = (uint8_t*) malloc(cm->vpw * cm->vph);

	f->predicted = (yuv_t*) malloc(sizeof(yuv_t));
	f->predicted->Y = (uint8_t*) calloc(cm->ypw * cm->yph, sizeof(uint8_t));
	f->predicted->U = (uint8_t*) calloc(cm->upw * cm->uph, sizeof(uint8_t));
	f->predicted->V = (uint8_t*) calloc(cm->vpw * cm->vph, sizeof(uint8_t));

	f->residuals = (dct_t*) malloc(sizeof(dct_t));
	f->residuals->Ydct = (int16_t*) calloc(cm->ypw * cm->yph, sizeof(int16_t));
	f->residuals->Udct = (int16_t*) calloc(cm->upw * cm->uph, sizeof(int16_t));
	f->residuals->Vdct = (int16_t*) calloc(cm->vpw * cm->vph, sizeof(int16_t));

	f->mbs[Y_COMPONENT] = (macroblock*) calloc(cm->mb_rows * cm->mb_cols,
			sizeof(struct macroblock));
	f->mbs[U_COMPONENT] = (macroblock*) calloc(cm->mb_rows / 2 * cm->mb_cols / 2,
			sizeof(struct macroblock));
	f->mbs[V_COMPONENT] = (macroblock*) calloc(cm->mb_rows / 2 * cm->mb_cols / 2,
			sizeof(struct macroblock));

	return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
	fwrite(image->Y, 1, w * h, fp);
	fwrite(image->U, 1, w * h / 4, fp);
	fwrite(image->V, 1, w * h / 4, fp);
}
