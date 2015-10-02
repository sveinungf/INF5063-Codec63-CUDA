#ifndef C63_DSP_H_
#define C63_DSP_H_

#define ISQRT2 0.70710678118654f


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

#include <inttypes.h>

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl);

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl);

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result);

void cuda_init();
void cuda_cleanup();

#endif  /* C63_DSP_H_ */
