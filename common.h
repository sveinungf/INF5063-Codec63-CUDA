#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include <inttypes.h>

#include "c63.h"
#include "c63_cuda.h"

// Declarations
struct frame* create_frame(struct c63_common *cm, const struct c63_cuda& c63_cuda);

yuv_t* create_image(struct c63_common *cm);

yuv_t* create_image_gpu(struct c63_common *cm);

void destroy_frame(struct frame *f);

void destroy_image(yuv_t* image);

void destroy_image_gpu(yuv_t* image);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

#endif  /* C63_COMMON_H_ */
