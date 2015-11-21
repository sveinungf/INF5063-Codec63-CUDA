#ifndef INIT_CUDA_H_
#define INIT_CUDA_H_

#include "c63.h"
#include "c63_cuda.h"

struct c63_cuda init_c63_cuda();
void cleanup_c63_cuda(struct c63_cuda& c63_cuda);

#endif /* INIT_CUDA_H_ */
