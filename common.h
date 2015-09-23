#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include <inttypes.h>

#include "c63.h"

// Declarations
struct frame* create_frame(struct c63_common *cm, yuv_t *image);

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, int16_t *out_data, uint8_t *quantization);

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
    uint32_t height, uint8_t *out_data, uint8_t *quantization);

void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

#endif  /* C63_COMMON_H_ */
