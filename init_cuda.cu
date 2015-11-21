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
