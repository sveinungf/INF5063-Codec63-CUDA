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

void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h, int y,
		uint8_t *out_data, uint8_t *quantization)
{
	int x;

	int16_t block[8 * 8];

	/* Perform the dequantization and iDCT */
	for (x = 0; x < w; x += 8)
	{
		int i, j;

		dequant_idct_block_8x8(in_data + (x * 8), block, quantization);

		for (i = 0; i < 8; ++i)
		{
			for (j = 0; j < 8; ++j)
			{
				/* Add prediction block. Note: DCT is not precise -
				 Clamp to legal values */
				int16_t tmp = block[i * 8 + j] + (int16_t) prediction[i * w + j + x];

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
	}
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		uint8_t *out_data, uint8_t *quantization)
{
	unsigned int y;

	for (y = 0; y < height; y += 8)
	{
		dequantize_idct_row(in_data + y * width, prediction + y * width, width, height, y,
				out_data + y * width, quantization);
	}
}

void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h, int16_t *out_data,
		uint8_t *quantization)
{
	int x;

	int16_t block[8 * 8];

	/* Perform the DCT and quantization */
	for (x = 0; x < w; x += 8)
	{
		int i, j;

		for (i = 0; i < 8; ++i)
		{
			for (j = 0; j < 8; ++j)
			{
				block[i * 8 + j] = ((int16_t) in_data[i * w + j + x] - prediction[i * w + j + x]);
			}
		}

		/* Store MBs linear in memory, i.e. the 64 coefficients are stored
		 continous. This allows us to ignore stride in DCT/iDCT and other
		 functions. */
		dct_quant_block_8x8(block, out_data + (x * 8), quantization);
	}
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		int16_t *out_data, uint8_t *quantization)
{
	unsigned int y;

	for (y = 0; y < height; y += 8)
	{
		dct_quantize_row(in_data + y * width, prediction + y * width, width, height,
				out_data + y * width, quantization);
	}
}

static void create_frame_gpu(struct c63_common* cm, struct frame* f)
{
	cudaMalloc((void**) &f->residuals->Ydct_gpu, cm->ypw * cm->yph * sizeof(int16_t));
	cudaMalloc((void**) &f->residuals->Udct_gpu, cm->upw * cm->uph * sizeof(int16_t));
	cudaMalloc((void**) &f->residuals->Vdct_gpu, cm->vpw * cm->vph * sizeof(int16_t));

	cudaMalloc((void**) &f->mbs_gpu[Y_COMPONENT], cm->mb_rowsY * cm->mb_colsY *
			sizeof(struct macroblock));
	cudaMalloc((void**) &f->mbs_gpu[U_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV *
			sizeof(struct macroblock));
	cudaMalloc((void**) &f->mbs_gpu[V_COMPONENT], cm->mb_rowsUV * cm->mb_colsUV *
			sizeof(struct macroblock));
}

static void destroy_frame_gpu(struct frame* f)
{
	cudaFree(f->residuals->Ydct_gpu);
	cudaFree(f->residuals->Udct_gpu);
	cudaFree(f->residuals->Vdct_gpu);

	cudaFree(f->mbs_gpu[Y_COMPONENT]);
	cudaFree(f->mbs_gpu[U_COMPONENT]);
	cudaFree(f->mbs_gpu[V_COMPONENT]);
}

struct frame* create_frame(struct c63_common *cm)
{
	struct frame *f = (frame*) malloc(sizeof(struct frame));

	f->recons = create_image(cm);
	f->predicted = create_image(cm);

	f->residuals = (dct_t*) malloc(sizeof(dct_t));
	f->residuals->Ydct = (int16_t*) malloc(cm->ypw * cm->yph * sizeof(int16_t));
	f->residuals->Udct = (int16_t*) malloc(cm->upw * cm->uph * sizeof(int16_t));
	f->residuals->Vdct = (int16_t*) malloc(cm->vpw * cm->vph * sizeof(int16_t));

	f->mbs[Y_COMPONENT] = (macroblock*) malloc(cm->mb_rowsY * cm->mb_colsY *
			sizeof(struct macroblock));
	f->mbs[U_COMPONENT] = (macroblock*) malloc(cm->mb_rowsUV * cm->mb_colsUV *
			sizeof(struct macroblock));
	f->mbs[V_COMPONENT] = (macroblock*) malloc(cm->mb_rowsUV * cm->mb_colsUV *
			sizeof(struct macroblock));

	create_frame_gpu(cm, f);

	return f;
}

void destroy_frame(struct frame *f)
{
	destroy_frame_gpu(f);

	destroy_image(f->recons);
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

static void create_image_gpu(struct c63_common* cm, yuv_t* image)
{
	cudaMalloc((void**) &image->Y_gpu, cm->ypw * cm->yph * sizeof(uint8_t));
	cudaMalloc((void**) &image->U_gpu, cm->upw * cm->uph * sizeof(uint8_t));
	cudaMalloc((void**) &image->V_gpu, cm->vpw * cm->vph * sizeof(uint8_t));
}

static void destroy_image_gpu(yuv_t* image)
{
	cudaFree(image->Y_gpu);
	cudaFree(image->U_gpu);
	cudaFree(image->V_gpu);
}

yuv_t* create_image(struct c63_common *cm)
{
	yuv_t* image = (yuv_t*) malloc(sizeof(yuv_t));
	image->Y = (uint8_t*) malloc(cm->ypw * cm->yph * sizeof(uint8_t));
	image->U = (uint8_t*) malloc(cm->upw * cm->uph * sizeof(uint8_t));
	image->V = (uint8_t*) malloc(cm->vpw * cm->vph * sizeof(uint8_t));

	create_image_gpu(cm, image);

	return image;
}

void destroy_image(yuv_t *image)
{
	destroy_image_gpu(image);

	free(image->Y);
	free(image->U);
	free(image->V);
	free(image);
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
	fwrite(image->Y, 1, w * h, fp);
	fwrite(image->U, 1, w * h / 4, fp);
	fwrite(image->V, 1, w * h / 4, fp);
}
