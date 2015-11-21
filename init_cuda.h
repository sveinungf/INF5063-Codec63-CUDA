#ifndef INIT_CUDA_H_
#define INIT_CUDA_H_

#include "c63.h"
#include "c63_cuda.h"

struct c63_cuda init_c63_cuda();
void cleanup_c63_cuda(struct c63_cuda& c63_cuda);

struct c63_common_gpu init_c63_gpu(const struct c63_common* cm, const struct c63_cuda& c63_cuda);
void cleanup_c63_gpu(struct c63_common_gpu& cm_gpu);

#endif /* INIT_CUDA_H_ */
