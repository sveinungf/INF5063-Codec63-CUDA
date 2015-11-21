#ifndef C63_ME_CUDA_H_
#define C63_ME_CUDA_H_

#include "c63.h"

namespace c63 {
namespace gpu {

template<int component>
void c63_motion_estimate(struct c63_common *cm, const struct c63_common_gpu& cm_gpu,
		const struct c63_cuda& c63_cuda);

template<int component>
void c63_motion_compensate(struct c63_common *cm, const struct c63_cuda& c63_cuda);

}
}

#endif  /* C63_ME_CUDA_H_ */
