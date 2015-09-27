#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

extern "C" {
#include "dsp.h"
#include "tables.h"
}


__constant__ float dct_lookup[64] = 
{
	1.00000000000000000000f, 0.98078525066375732422f, 0.92387950420379638672f, 0.83146959543228149414f, 0.70710676908493041992f, 0.55557024478912353516f, 0.38268342614173889160f, 0.19509032368659973145f,
	1.00000000000000000000f, 0.83146959543228149414f, 0.38268342614173889160f,-0.19509032368659973145f,-0.70710676908493041992f,-0.98078525066375732422f,-0.92387950420379638672f,-0.55557024478912353516f,
	1.00000000000000000000f, 0.55557024478912353516f,-0.38268342614173889160f,-0.98078525066375732422f,-0.70710676908493041992f, 0.19509032368659973145f, 0.92387950420379638672f, 0.83146959543228149414f,
	1.00000000000000000000f, 0.19509032368659973145f,-0.92387950420379638672f,-0.55557024478912353516f, 0.70710676908493041992f, 0.83146959543228149414f,-0.38268342614173889160f,-0.98078525066375732422f,
	1.00000000000000000000f,-0.19509032368659973145f,-0.92387950420379638672f, 0.55557024478912353516f, 0.70710676908493041992f,-0.83146959543228149414f,-0.38268342614173889160f, 0.98078525066375732422f,
	1.00000000000000000000f,-0.55557024478912353516f,-0.38268342614173889160f, 0.98078525066375732422f,-0.70710676908493041992f,-0.19509032368659973145f, 0.92387950420379638672f,-0.83146959543228149414f,
	1.00000000000000000000f,-0.83146959543228149414f, 0.38268342614173889160f, 0.19509032368659973145f,-0.70710676908493041992f, 0.98078525066375732422f,-0.92387950420379638672f, 0.55557024478912353516f,
	1.00000000000000000000f,-0.98078525066375732422f, 0.92387950420379638672f,-0.83146959543228149414f, 0.70710676908493041992f,-0.55557024478912353516f, 0.38268342614173889160f,-0.19509032368659973145f
};



float *gpu_in, *gpu_out;

__host__ void cuda_init() {
	
	cudaMalloc(&gpu_in, 64*sizeof(float));
	cudaMalloc(&gpu_out, 64*sizeof(float));	
}

__host__ void cuda_cleanup() {
	cudaFree(gpu_in);
	cudaFree(gpu_out);
}


__global__ void gpu_transpose_block(float *in_data, float *out_data) 
{
	
    int i = threadIdx.y;
    int j = threadIdx.x;
    out_data[i * 8 + j] = in_data[j * 8 + i];
    /*
    int j;
    for(j = 0; j < 8; j++) {	
		out_data[i * 8 + j] = in_data[j * 8 + i];
	}*/
}

__host__ void transpose_block(float *in_data, float *out_data)
{
	gpu_transpose_block<<<1, 8>>>(gpu_in, gpu_out);
	
	
	
	/*
	 * 
	 * float* d_A; 
	 * cudaMalloc(&d_A, size);
	 * 
	 * float* d_B; 
	 * cudaMalloc(&d_B, size); 
	 * 
	 * float* d_C; 
	 * cudaMalloc(&d_C, size); 
	 * 
	 * // Copy vectors from host memory to device memory 
	 * cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
	 * cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); 
	 * 
	 * // Invoke kernel 
	 * int threadsPerBlock = 256; 
	 * int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
	 * VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); 
	 * 
	 * // Copy result from device memory to host memory 
	 * // h_C contains the result in host memory 
	 * cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); 
	 * 
	 * // Free device memory 
	 * cudaFree(d_A); 
	 * cudaFree(d_B); 
	 * cudaFree(d_C);
*/
	
	/*
	int i, j;
	

	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			out_data[i * 8 + j] = in_data[j * 8 + i];
		}
	}
	* */
	
}

/*
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
* */

__global__ void gpu_dct_1d(float *in_data, float *out_data)
{
	
	int i = threadIdx.y;
	int j = threadIdx.x;
	
	out_data[i] += in_data[j] * dct_lookup[j*8+i];
	
	/*int i = threadIdx.x;
	out_data[i/8] += in_data[i/8] * dct_lookup[i];*/
}

/*
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
* */

__global__ void gpu_idct_1d(float *in_data, float *out_data)
{
	int i, j;
	i = threadIdx.x;
		
	float idct = 0;

	for (j = 0; j < 8; ++j)
	{
		idct += in_data[j] * dct_lookup[i*8+j];
	}

	out_data[i] = idct;	
	
}


static void scale_block(float *in_data, float *out_data)
{
	int u, v;

	for (v = 0; v < 8; ++v)
	{
		for (u = 0; u < 8; ++u)
		{
			float a1 = !u ? ISQRT2 : 1.0f;
			float a2 = !v ? ISQRT2 : 1.0f;

			/* Scale according to normalizing function */
			out_data[v * 8 + u] = in_data[v * 8 + u] * a1 * a2;
		}
	}
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
	int zigzag;

	for (zigzag = 0; zigzag < 64; ++zigzag)
	{
		uint8_t u = zigzag_U[zigzag];
		uint8_t v = zigzag_V[zigzag];

		float dct = in_data[v * 8 + u];

		/* Zig-zag and quantize */
		out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
	}
}

static void dequantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
	int zigzag;

	for (zigzag = 0; zigzag < 64; ++zigzag)
	{
		uint8_t u = zigzag_U[zigzag];
		uint8_t v = zigzag_V[zigzag];

		float dct = in_data[zigzag];

		/* Zig-zag and de-quantize */
		out_data[v * 8 + u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
	}
}

__host__ void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
	float mb[8 * 8];
	float mb2[8 * 8];

	int i, v;

	for (i = 0; i < 64; ++i)
	{
		mb2[i] = in_data[i];
	}
	
	int numBlocks = 1;
	dim3 threadsPerBlock(8, 8);
	
	
	cudaMemcpy(gpu_in, (float*)&mb2, 64*sizeof(float), cudaMemcpyHostToDevice); 

	/* Two 1D DCT operations with transpose */
	
	for (v = 0; v < 8; ++v)
	{
		//dct_1d(mb2 + v * 8, mb + v * 8);
		//gpu_dct_1d<<<numBlocks, 8>>>(gpu_in + v * 8, gpu_out + v * 8);
		gpu_dct_1d<<<numBlocks, threadsPerBlock>>>(gpu_in + v * 8, gpu_out + v * 8);
	}

	//transpose_block(mb, mb2);
	gpu_transpose_block<<<numBlocks, threadsPerBlock>>>(gpu_out, gpu_in);
	
	for (v = 0; v < 8; ++v)
	{
		//dct_1d(mb2 + v * 8, mb + v * 8);
		//gpu_dct_1d<<<numBlocks, 8>>>(gpu_in + v * 8, gpu_out + v * 8);
		gpu_dct_1d<<<numBlocks, threadsPerBlock>>>(gpu_in + v * 8, gpu_out + v * 8);
	}
	
	gpu_transpose_block<<<numBlocks, threadsPerBlock>>>(gpu_out, gpu_in);
	//transpose_block(mb, mb2);
	
	cudaMemcpy((float*)&mb2, gpu_in, 64*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((float*)&mb, gpu_out, 64*sizeof(float), cudaMemcpyDeviceToHost);

	scale_block(mb2, mb);
	quantize_block(mb, mb2, quant_tbl);

	for (i = 0; i < 64; ++i)
	{
		out_data[i] = mb2[i];
	}
}

__host__ void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
	float mb[8 * 8];
	float mb2[8 * 8];

	int i, v;

	for (i = 0; i < 64; ++i)
	{
		mb[i] = in_data[i];
	}

	dequantize_block(mb, mb2, quant_tbl);
	scale_block(mb2, mb);
	
	cudaMemcpy(gpu_in, (float*)&mb, 64*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(gpu_out, (float*)&mb2, 64*sizeof(float), cudaMemcpyHostToDevice);
	
	int numBlocks = 1;
	dim3 threadsPerBlock(8, 8);

	/* Two 1D inverse DCT operations with transpose */
	for (v = 0; v < 8; ++v)
	{
		//idct_1d(mb + v * 8, mb2 + v * 8);
		gpu_idct_1d<<<numBlocks, 8>>>(gpu_in + v * 8, gpu_out + v * 8);
	}
	//transpose_block(mb2, mb);
	gpu_transpose_block<<<numBlocks, threadsPerBlock>>>(gpu_out, gpu_in);
	for (v = 0; v < 8; ++v)
	{
		//idct_1d(mb + v * 8, mb2 + v * 8);
		gpu_idct_1d<<<numBlocks, 8>>>(gpu_in + v * 8, gpu_out + v * 8);
	}
	//transpose_block(mb2, mb);
	gpu_transpose_block<<<numBlocks, threadsPerBlock>>>(gpu_out, gpu_in);

	
	cudaMemcpy((float*)&mb, gpu_in, 64*sizeof(float), cudaMemcpyDeviceToHost);
	
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
