#include "init_cuda.h"

struct c63_cuda init_c63_cuda()
{
	struct c63_cuda result;

	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		cudaStreamCreate(&result.stream[c]);
	}

	return result;
}

void cleanup_c63_cuda(struct c63_cuda& c63_cuda)
{
	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		cudaStreamDestroy(c63_cuda.stream[c]);
	}
}

struct c63_common_gpu init_c63_gpu(const struct c63_common* cm, const struct c63_cuda& c63_cuda)
{
	struct c63_common_gpu result;

	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		int cols = cm->mb_cols[c];
		int rows = cm->mb_rows[c];

		cudaMalloc(&result.sad_index_results[c], cols * rows * sizeof(unsigned int));
	}

	return result;
}

void cleanup_c63_gpu(struct c63_common_gpu& cm_gpu)
{
	for (int c = 0; c < COLOR_COMPONENTS; ++c)
	{
		cudaFree(cm_gpu.sad_index_results[c]);
	}
}
