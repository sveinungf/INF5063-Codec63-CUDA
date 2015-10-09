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

