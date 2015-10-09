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

int16_t *gpu_temp_16[COLOR_COMPONENTS];

int16_t *gpu_out_16[COLOR_COMPONENTS];
//uint8_t *gpu_out_8;


__constant__ uint32_t gpu_offsets[64];

void cuda_init(struct c63_common cm){

	cudaMalloc(&gpu_temp_16[Y_COMPONENT], cm.padw[Y_COMPONENT]*cm.padh[Y_COMPONENT]*sizeof(int16_t));
	cudaMalloc(&gpu_temp_16[U_COMPONENT], cm.padw[U_COMPONENT]*cm.padh[U_COMPONENT]*sizeof(int16_t));
	cudaMalloc(&gpu_temp_16[V_COMPONENT], cm.padw[V_COMPONENT]*cm.padh[V_COMPONENT]*sizeof(int16_t));

	cudaMalloc(&gpu_out_16[Y_COMPONENT], cm.padw[Y_COMPONENT]*cm.padh[Y_COMPONENT]*sizeof(int16_t));
	cudaMalloc(&gpu_out_16[U_COMPONENT], cm.padw[U_COMPONENT]*cm.padh[U_COMPONENT]*sizeof(int16_t));
	cudaMalloc(&gpu_out_16[V_COMPONENT], cm.padw[V_COMPONENT]*cm.padh[V_COMPONENT]*sizeof(int16_t));

	//cudaMalloc(&gpu_out_16, width*height*sizeof(int16_t));
	//cudaMalloc(&gpu_out_8, width*height*sizeof(uint8_t));

	/*uint32_t offsets[64];
	int i, j;
	for (i = 0; i < 8; ++i) {
		for(j = 0; j < 8; ++j) {
			offsets[i*8+j] = i * width + j;
		}
	}
	cudaMemcpyToSymbol(gpu_offsets, offsets, 64*sizeof(uint32_t));*/
}

void cuda_cleanup() {
	cudaFree(gpu_temp_16[Y_COMPONENT]);
	cudaFree(gpu_temp_16[U_COMPONENT]);
	cudaFree(gpu_temp_16[V_COMPONENT]);

	cudaFree(gpu_out_16[Y_COMPONENT]);
	cudaFree(gpu_out_16[U_COMPONENT]);
	cudaFree(gpu_out_16[V_COMPONENT]);

	//cudaFree(gpu_out_16);
	//cudaFree(gpu_out_8);
}

__global__
void dequantize_add_pred(int16_t *in_data, uint8_t *prediction, int w, uint8_t *out_data)
{
	int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

	int x_offset = blockIdx.x * 8;
	int y_offset = blockIdx.y * w*8;

	int i = threadIdx.y;
	int j = threadIdx.x;

	__shared__ float idct_in[64];

	// Add prediction block. Note: DCT is not precise -
	// Clamp to legal values
	idct_in[i*8+j] = in_data[block_offset + i * 8 + j] + (int16_t) prediction[i * w + j + x_offset + y_offset];
	if (idct_in[i*8+j] < 0)
	{
		idct_in[i*8+j] = 0;
	}
	else if (idct_in[i*8+j] > 255)
	{
		idct_in[i*8+j] = 255;
	}

	out_data[i * w + j + x_offset + y_offset] = idct_in[i*8+j];
}

/*
__device__
void test_pred(int16_t *in_data, uint8_t *out_data, uint8_t *prediction, int x_offset, int y_offset, int w, int i, int j)
{
	// Add prediction block. Note: DCT is not precise -
	// Clamp to legal values
	int16_t tmp = in_data[i * 8 + j] + (int16_t) prediction[i * w + j + x_offset + y_offset];
	if (tmp < 0)
	{
		tmp = 0;
	}
	else if (tmp > 255)
	{
		tmp = 255;
	}

	out_data[i * w + j + x_offset + y_offset] = tmp;

}
*/
__global__ void dequantize_idct(int16_t *in_data, uint8_t *prediction, int w, uint8_t *out_data, int quantization)
{
	int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

 	int i = threadIdx.y;
 	int j = threadIdx.x;

 	__shared__ float idct_in[64];
 	__shared__ float idct_out[64];

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

/*
__host__ void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		uint8_t *out_data, int quantization, cudaStream_t stream)
{
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	dequantize_idct_blocks<<<numBlocks, threadsPerBlock, 0, stream>>>(in_data, prediction, width, out_data, quantization);
}
*/

__global__ void dct_quantize(uint8_t *in_data, uint8_t *prediction, int w, int16_t *out_data, int quantization)
{
	int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

	int i = threadIdx.y;
	int j = threadIdx.x;

	int offset = blockIdx.x * 8 + blockIdx.y * w*8 + i*w+j;

	__shared__ float dct_in[64];
	__shared__ float dct_out[64];

	dct_in[i*8+j] = ((float) in_data[offset] - prediction[offset]);
	__syncthreads();
	dct_quant_block_8x8(dct_in, dct_out, quantization, i, j);

	out_data[block_offset + i*8+j] = dct_in[i*8+j];
}
/*
__global__ void dct_quantize_row3(uint8_t *in_data, uint8_t *prediction, int w, int h, int16_t *out_data,
		int quantization, uint32_t blocksPerRow, uint32_t pixelsPerRow)
{
	int block_id = blockIdx.x + blockIdx.y * gridDim.x;
	int block_offset = block_id * blockDim.x * blockDim.y;

	int x_offset = (block_id % gridDim.x) * 8;
	int offset = (block_id / gridDim.x) * pixelsPerRow;

	int i = threadIdx.y;
	int j = threadIdx.x;

	dct_temp[i * 8 + j] = ((int16_t) in_data[i * w + j + offset + x_offset] - prediction[i * w + j + offset + x_offset]);
	__syncthreads();

	dct_quant_block_8x8(dct_temp, out_data + block_offset, quantization, i, j);
}
*/
__host__ void dct_quantize3(struct c63common *cm, int16_t *gpu_Y_out, int16_t *gpu_U_out, int16_t *gpu_V_out)
{	/*
	 cudaMemcpy(gpu_Y_pred, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT]*sizeof(uint8_t), cudaMemcpyHostToDevice);
	 cudaMemcpy(gpu_U_pred, cm->curframe->predicted->U, cm->padw[U_COMPONENT]*cm->padh[U_COMPONENT]*sizeof(uint8_t), cudaMemcpyHostToDevice);
	 cudaMemcpy(gpu_V_pred, cm->curframe->predicted->V, cm->padw[V_COMPONENT]*cm->padh[V_COMPONENT]*sizeof(uint8_t), cudaMemcpyHostToDevice);

	 cudaMemcpy(gpu_Y_8, image->Y, cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT]*sizeof(uint8_t), cudaMemcpyHostToDevice);
	 cudaMemcpy(gpu_U_8, image->U, cm->padw[U_COMPONENT]*cm->padh[U_COMPONENT]*sizeof(uint8_t), cudaMemcpyHostToDevice);
	 cudaMemcpy(gpu_V_8, image->V, cm->padw[V_COMPONENT]*cm->padh[V_COMPONENT]*sizeof(uint8_t), cudaMemcpyHostToDevice);

	 dct_quantize(gpu_Y_8, gpu_Y_pred, cm->padw[Y_COMPONENT],
	     cm->padh[Y_COMPONENT], gpu_Y_16, cm->curframe->residuals->Ydct,
	     Y_COMPONENT);

	 dct_quantize(gpu_U_8, gpu_U_pred, cm->padw[U_COMPONENT],
	     cm->padh[U_COMPONENT], gpu_U_16, cm->curframe->residuals->Udct,
	     U_COMPONENT);

	 dct_quantize(gpu_V_8, gpu_V_pred, cm->padw[V_COMPONENT],
	     cm->padh[V_COMPONENT], gpu_V_16, cm->curframe->residuals->Vdct,
	     V_COMPONENT);

	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	uint32_t blocksPerRow = width/8;
	uint32_t pixelsPerRow = width*8;

	unsigned int y;
	dct_quantize_row<<<numBlocks, threadsPerBlock>>>(gpu_Y_8, gpu_Y_pred, cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT],
				gpu_Y_out, Y_COMPONENT, gpu_U_8, gpu_U_pred, cm->padw[U_COMPONENT], cm->padh[U_COMPONENT],
				gpu_U_out, U_COMPONENT, gpu_V_8, gpu_V_pred, cm->padw[V_COMPONENT], cm->padh[V_COMPONENT],
				gpu_V_out, V_COMPONENT, blocksPerRow, pixelsPerRow);

	cudaMemcpy(cm->curframe->residuals->Ydct, gpu_Y_out, cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT]*sizeof(int16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(cm->curframe->residuals->Udct, gpu_U_out, cm->padw[U_COMPONENT]*cm->padh[U_COMPONENT]*sizeof(int16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(cm->curframe->residuals->Vdct, gpu_V_out, cm->padw[V_COMPONENT]*cm->padh[V_COMPONENT]*sizeof(int16_t), cudaMemcpyDeviceToHost);




	//dct_quantize(gpu_Y_8, gpu_Y_pred, cm->padw[Y_COMPONENT],
    //cm->padh[Y_COMPONENT], gpu_Y_16, cm->curframe->residuals->Ydct,
    //Y_COMPONENT);
     * */
}

/*
__host__ void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		int16_t *out_data, int quantization, cudaStream_t stream)
{
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	dct_quantize_blocks<<<numBlocks, threadsPerBlock, 0, stream>>>(in_data, prediction, width,
				out_data, quantization);
}
*/

void dequantize_idct_row2(int16_t *in_data, uint8_t *prediction, int w, int h, int y,
		uint8_t *out_data, uint8_t *quantization)
{
	int x;

	int16_t block[8 * 8];

	/* Perform the dequantization and iDCT */
	for (x = 0; x < w; x += 8)
	{
		int i, j;

		//dequant_idct_block_8x8(in_data + (x * 8), block, quantization);

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

__host__ void dequantize_idct2(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		uint8_t *out_data, uint8_t *quantization)
{
	unsigned int y;

	for (y = 0; y < height; y += 8)
	{
		dequantize_idct_row2(in_data + y * width, prediction + y * width, width, height, y,
				out_data + y * width, quantization);
	}
}

__global__ void dct_quantize_row2(uint8_t *in_data, uint8_t *prediction, int w, int h, int16_t *out_data,
		int quantization)
{
	/* Perform the DCT and quantization */
//	int x;
	//for (x = 0; x < w; x += 8)
	//{
	float block[8 * 8];

	int block_offset = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

	int offset = blockIdx.x * 8 + blockIdx.y * w*8;

	int i, j;

	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			block[i * 8 + j] = ((float) in_data[offset + i * w + j] - prediction[offset + i * w + j]);
		}
	}

	/* Store MBs linear in memory, i.e. the 64 coefficients are stored
	 continous. This allows us to ignore stride in DCT/iDCT and other
	 functions. */
	dct2_quant_block_8x8(block, out_data + block_offset, quantization);
	//}
}

__host__ void dct_quantize2(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		int16_t *gpu_out_data, int16_t *out_data, int quantization)
{
	dim3 threadsPerBlock(8,8);
	dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

	//for (y = 0; y < height; y += 8)
	//{
	dct_quantize_row2<<<numBlocks, threadsPerBlock>>>(in_data, prediction, width, height,
				gpu_out_data, quantization);
	//}
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

	cudaFree(f->residuals->Ydct);
	cudaFree(f->residuals->Udct);
	cudaFree(f->residuals->Vdct);
	free(f->residuals);

	cudaFree(f->mbs[Y_COMPONENT]);
	cudaFree(f->mbs[U_COMPONENT]);
	cudaFree(f->mbs[V_COMPONENT]);

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
