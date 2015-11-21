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

static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

static void init_frame_gpu(struct c63_common* cm, struct frame* f)
{
	f->recons_gpu = create_image_gpu(cm);
	f->predicted_gpu = create_image_gpu(cm);

	f->residuals_gpu = (dct_t*) malloc(sizeof(dct_t));
	cudaMalloc((void**) &f->residuals_gpu->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
	cudaMalloc((void**) &f->residuals_gpu->Udct, cm->upw * cm->uph * sizeof(int16_t));
	cudaMalloc((void**) &f->residuals_gpu->Vdct, cm->vpw * cm->vph * sizeof(int16_t));

	cudaMalloc((void**) &f->mbs_gpu[Y_COMPONENT],
			cm->mb_rows[Y] * cm->mb_cols[Y] * sizeof(struct macroblock));
	cudaMalloc((void**) &f->mbs_gpu[U_COMPONENT],
			cm->mb_rows[U] * cm->mb_cols[U] * sizeof(struct macroblock));
	cudaMalloc((void**) &f->mbs_gpu[V_COMPONENT],
			cm->mb_rows[U] * cm->mb_cols[U] * sizeof(struct macroblock));
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

struct macroblock *create_mb(struct macroblock *mb, size_t size, const cudaStream_t& stream)
{
	cudaMallocHost((void**) &mb, size);
	cudaMemsetAsync(mb, 0, size, stream);

	return mb;
}

struct frame* create_frame(struct c63_common *cm, const struct c63_cuda& c63_cuda)
{
	struct frame *f = (frame*) malloc(sizeof(struct frame));

	f->residuals = (dct_t*) malloc(sizeof(dct_t));
	cudaMallocHost((void**) &f->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
	cudaMallocHost((void**) &f->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
	cudaMallocHost((void**) &f->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));

	size_t sizeY = cm->mb_rows[Y] * cm->mb_cols[Y] * sizeof(struct macroblock);
	size_t sizeUV = cm->mb_rows[U] * cm->mb_cols[U] * sizeof(struct macroblock);
	f->mbs[Y_COMPONENT] = create_mb(f->mbs[Y_COMPONENT], sizeY, c63_cuda.stream[Y]);
	f->mbs[U_COMPONENT] = create_mb(f->mbs[U_COMPONENT], sizeUV, c63_cuda.stream[U]);
	f->mbs[V_COMPONENT] = create_mb(f->mbs[V_COMPONENT], sizeUV, c63_cuda.stream[V]);

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
	cudaHostAlloc((void**) &image->Y, cm->ypw * cm->yph * sizeof(uint8_t),
			cudaHostAllocWriteCombined);
	cudaHostAlloc((void**) &image->U, cm->upw * cm->uph * sizeof(uint8_t),
			cudaHostAllocWriteCombined);
	cudaHostAlloc((void**) &image->V, cm->vpw * cm->vph * sizeof(uint8_t),
			cudaHostAllocWriteCombined);

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
