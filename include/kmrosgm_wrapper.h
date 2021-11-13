#ifndef __LIBSGM_WRAPPER_H__
#define __LIBSGM_WRAPPER_H__

#include "kmrosgm.h"
#include <memory>
#ifdef BUILD_OPENCV_WRAPPER
#include <opencv2/core/cuda.hpp>
#endif

namespace sgm {

class LibSGMWrapper {
public:
    int a;
};

}

#endif // __LIBSGM_WRAPPER_H__