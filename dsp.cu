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

__device__ void dct_quant_block_8x8(float* in_data, float *out_data, int quant_index, int i, int j)
{
	// First dct_1d - mb = mb2 - and transpose
	float dct = 0;
	int k;
	for (k = 0; k < 8; ++k) {
		dct += in_data[j*8+k] * dct_lookup[k*8+i];
	}
	__syncthreads();
	in_data[i*8+j] = dct;
	__syncthreads();

	// Second dct_1d - mb = mb2 - and transpose
	dct = 0;
	for (k = 0; k < 8; ++k) {
		dct += in_data[j*8+k] * dct_lookup[k*8+i];
	}

	// Scale
	out_data[i*8+j] = dct * a1[i*8+j] * a2[i*8+j];
	__syncthreads();

	// Quantize and set value in out_data
	dct = out_data[UV_indexes[i*8+j]];
	in_data[i*8+j] = (float) round((dct/4.0) / quant_table[quant_index*64 + i*8+j]);
}


__device__ void dequant_idct_block_8x8(float *in_data, float *out_data, int quant_index, int i, int j)
{
	// Dequantize
	float dct = in_data[i*8+j];
	out_data[UV_indexes[i*8+j]] = (float) round((dct*quant_table[quant_index*64 + i*8+j]) / 4.0f);
	__syncthreads();

	// Scale
	in_data[i*8+j] = out_data[i*8+j] * a1[i*8+j] * a2[i*8+j];
	__syncthreads();

	// First idct - mb2 = mb - and transpose
	float idct = 0;
	int k;
	for (k = 0; k < 8; ++k) {
		idct += in_data[j*8+k] * dct_lookup[i*8+k];
	}
	__syncthreads();
	in_data[i*8+j] = idct;
	__syncthreads();

	// Second idct - mb2 = mb - and transpose
	idct = 0;
	for (k = 0; k < 8; ++k) {
		idct += in_data[j*8+k] * dct_lookup[i*8+k];
	}
	out_data[i*8+j] = idct;
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


__device__ static void transpose_block(float *in_data, float *out_data)
{
	int i, j;

	for (i = 0; i < 8; ++i)
	{
		for (j = 0; j < 8; ++j)
		{
			out_data[i * 8 + j] = in_data[j * 8 + i];
		}
	}
}

__device__ static void dct_1d(float *in_data, float *out_data)
{
	int i, j;

	for (i = 0; i < 8; ++i)
	{
		float dct = 0;

		for (j = 0; j < 8; ++j)
		{
			dct += in_data[j] * dct_lookup[j*8+i];
		}

		out_data[i] = dct;
	}
}

__device__ static void idct_1d(float *in_data, float *out_data)
{
	int i, j;

	for (i = 0; i < 8; ++i)
	{
		float idct = 0;

		for (j = 0; j < 8; ++j)
		{
			idct += in_data[j] * dct_lookup[i*8+j];
		}

		out_data[i] = idct;
	}
}

__device__ static void scale_block(float *in_data, float *out_data)
{
	in_data[0] *= ISQRT2 * ISQRT2;

	in_data[1] *= ISQRT2;
	in_data[2] *= ISQRT2;
	in_data[3] *= ISQRT2;
	in_data[4] *= ISQRT2;
	in_data[5] *= ISQRT2;
	in_data[6] *= ISQRT2;
	in_data[7] *= ISQRT2;

	in_data[8] *= ISQRT2;
	in_data[16] *= ISQRT2;
	in_data[24] *= ISQRT2;
	in_data[32] *= ISQRT2;
	in_data[40] *= ISQRT2;
	in_data[48] *= ISQRT2;
	in_data[56] *= ISQRT2;
}

__device__ static void quantize_block(float *in_data, float *out_data, int quant_index)
{
	int zigzag;

	for (zigzag = 0; zigzag < 64; ++zigzag)
	{
		float dct = in_data[UV_indexes[zigzag]];

		/* Zig-zag and quantize */
		out_data[zigzag] = (float) round((dct / 4.0) / quant_table[quant_index*64 + zigzag]);
	}
}

__device__ static void dequantize_block(float *in_data, float *out_data, int quant_index)
{
	int zigzag;

	for (zigzag = 0; zigzag < 64; ++zigzag)
	{
		// Dequantize
		float dct = in_data[zigzag];

		/* Zig-zag and de-quantize */
		out_data[UV_indexes[zigzag]] = (float) round((dct * quant_table[quant_index*64 + zigzag]) / 4.0);
	}
}

__device__ void dct2_quant_block_8x8(float *in_data, int16_t *out_data, int quant_index)
{
	float mb[8 * 8] __attribute((aligned(16)));
	//float mb2[8 * 8] __attribute((aligned(16)));

	int i, v;
	/*
	for (i = 0; i < 64; ++i)
	{
		mb2[i] = in_data[i];
	}
	 */
	/* Two 1D DCT operations with transpose */
	for (v = 0; v < 8; ++v)
	{
		dct_1d(in_data + v * 8, mb + v * 8);
	}
	transpose_block(mb, in_data);
	for (v = 0; v < 8; ++v)
	{
		dct_1d(in_data + v * 8, mb + v * 8);
	}
	transpose_block(mb, in_data);

	scale_block(in_data, mb);
	quantize_block(in_data, mb, quant_index);

	for (i = 0; i < 64; ++i)
	{
		out_data[i] = mb[i];
	}
}

__device__ void dequant2_idct_block_8x8(int16_t *in_data, int16_t *out_data, int quant_index)
{
	float mb[8 * 8] __attribute((aligned(16)));
	float mb2[8 * 8] __attribute((aligned(16)));

	int i, v;

	for (i = 0; i < 64; ++i)
	{
		mb[i] = in_data[i];
	}

	dequantize_block(mb, mb2, quant_index);
	scale_block(mb2, mb);

	/* Two 1D inverse DCT operations with transpose */
	for (v = 0; v < 8; ++v)
	{
		idct_1d(mb + v * 8, mb2 + v * 8);
	}
	transpose_block(mb2, mb);
	for (v = 0; v < 8; ++v)
	{
		idct_1d(mb + v * 8, mb2 + v * 8);
	}
	transpose_block(mb2, mb);

	for (i = 0; i < 64; ++i)
	{
		out_data[i] = mb[i];
	}
}
