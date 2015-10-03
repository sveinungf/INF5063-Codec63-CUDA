#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "dsp.h"

extern "C" {
	#include "tables.h"
}

__constant__ uint8_t quant_table[192] =
{
	6,  4,  4,  5,  4,  4,  6,  5,
	5,  5,  7,  6,  6,  7,  9,  16,
	10,  9,  8,  8,  9,  19,  14,  14,
	11,  16,  23,  20,  24,  12,  22,  20,
	22,  22,  25,  28,  36,  31,  25,  27,
	34,  27,  22,  22,  32,  43,  32,  34,
	38,  39,  41,  41,  41,  24,  30,  45,
	48,  44,  40,  48,  36,  40,  41,  39,
	/*************************************/
	6,  7,  7,  9,  8,  9,  18,  10,
	10,  18,  39,  26,  22,  26,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	/*************************************/
	6,  7,  7,  9,  8,  9,  18,  10,
	10,  18,  39,  26,  22,  26,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39,
	39,  39,  39,  39,  39,  39,  39,  39
};


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


__constant__ float a1[64] =
{
	ISQRT2,   1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	ISQRT2,   1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	ISQRT2,   1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	ISQRT2,   1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	ISQRT2,   1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	ISQRT2,   1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	ISQRT2,   1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	ISQRT2,   1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f
};

__constant__ float a2[64] =
{
	ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2,
	  1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	  1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	  1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	  1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	  1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	  1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,
	  1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f,   1.0f,  1.0f
};

/*
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
};*/


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


__global__ void gpu_dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, int quant_index)
{
	int i = threadIdx.y;
	int j = threadIdx.x;

	// Copy pixel to shared memory
	dct_macro_block2[i*8+j] = in_data[i*8+j];
	__syncthreads();
	

	// First dct_1d - mb = mb2
	float dct = 0;
	int k;	
	for (k = 0; k < 8; ++k) {
		dct += dct_macro_block2[i*8+k] * dct_lookup[k*8+j];
	}
	dct_macro_block[i*8+j] = dct;
	__syncthreads();
	
	
	// First transpose - mb2 = mb
	dct_macro_block2[i*8+j] = dct_macro_block[j*8+i];
	__syncthreads();
	
	
	// Second dct_1d - mb = mb2
	dct = 0;
	for (k = 0; k < 8; ++k) {
		dct += dct_macro_block2[i*8+k] * dct_lookup[k*8+j];
	}
	dct_macro_block[i*8+j] = dct;
	__syncthreads();
	
	
	// Second transpose - mb2 = mb
	dct_macro_block2[i*8+j] = dct_macro_block[j*8+i];
	__syncthreads();

	// Scale
	dct_macro_block[i*8+j] = dct_macro_block2[i*8+j] * a1[i*8+j] * a2[i*8+j];
	__syncthreads();

	// Quantize	
	dct = dct_macro_block[UV_indexes[i*8+j]];
	dct_macro_block2[i*8+j] = (float) round((dct/4.0) / quant_table[quant_index*64 + i*8+j]);
	
	// Set value in cuda_out_data
	out_data[i*8+j] = dct_macro_block2[i*8+j];
}


__host__ void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, int quant_index)
{
	// Copy in_data to gpu memory
	//cudaMemcpy(cuda_in_data, in_data , 64*sizeof(int16_t), cudaMemcpyHostToDevice);
	
	// Set number of blocks and number of threads per block
	int numBlocks = 1;
	dim3 threadsPerBlock(8, 8);
	
	// Call gpu kernel - one thread works on one pixel
	gpu_dct_quant_block_8x8<<<numBlocks, threadsPerBlock>>>(in_data, out_data, quant_index);

	// Copy out_data from gpu memory to host memory
	//cudaMemcpy(out_data, cuda_out_data, 64*sizeof(int16_t), cudaMemcpyDeviceToHost);
}



__global__ void gpu_dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, int quant_index)
{
	int i = threadIdx.y;
	int j = threadIdx.x;

	// Copy to shared memory
	idct_macro_block[i*8+j] = in_data[i*8+j];
	__syncthreads();

	
	// Dequantize
	float dct = idct_macro_block[i*8+j];
	idct_macro_block2[UV_indexes[i*8+j]] = (float) round((dct*quant_table[quant_index*64 + i*8+j]) / 4.0f);
	__syncthreads();
	
	// Scale
	idct_macro_block[i*8+j] = idct_macro_block2[i*8+j] * a1[i*8+j] * a2[i*8+j];
	__syncthreads();
		
	// First idct - mb2 = mb
	float idct = 0;
	int k;
	for (k = 0; k < 8; ++k) {
		idct += idct_macro_block[i*8+k] * dct_lookup[j*8+k];
	}
	idct_macro_block2[i*8+j] = idct;
	__syncthreads();
	

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
	idct_macro_block[i*8+j] = idct_macro_block2[j*8+i];
	
	// Copy to out_data
	out_data[i*8+j] = idct_macro_block[i*8+j];
}



__host__ void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, int quant_index)
{
	// Copy in_data to gpu memory
	//cudaMemcpy(cuda_in_data, in_data, 64*sizeof(int16_t), cudaMemcpyHostToDevice);

	// Set number of blocks and number of threads per block
	int numBlocks = 1;
	dim3 threadsPerBlock(8, 8);

	// Call gpu kernel - one thread works on one pixel
	gpu_dequant_idct_block_8x8<<<numBlocks, threadsPerBlock>>>(in_data, out_data, quant_index);

	// Copy out_data from gpu memory to host memory
	//cudaMemcpy(out_data, cuda_out_data, 64*sizeof(int16_t), cudaMemcpyDeviceToHost);
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
