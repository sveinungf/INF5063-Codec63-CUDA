#ifndef C63_DSP_H_
#define C63_DSP_H_

#define ISQRT2 0.70710678118654f

#include <inttypes.h>

__device__
void dct_quant_block_8x8(float *in_data, float *out_data, int quant_index, int i, int j);

__device__
void dequant_idct_block_8x8(float *in_data, float *out_data, int quant_index, int i, int j);

#endif  /* C63_DSP_H_ */
