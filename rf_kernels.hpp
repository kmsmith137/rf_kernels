#include "immintrin.h"

#ifndef _RF_KERNELS_HPP
#define _RF_KERNELS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "rf_kernels/core.hpp"
#include "rf_kernels/upsample.hpp"
#include "rf_kernels/downsample.hpp"
#include "rf_kernels/xorshift_plus.hpp"
#include "rf_kernels/online_mask_filler.hpp"
#include "rf_kernels/polynomial_detrender.hpp"
#include "rf_kernels/spline_detrender.hpp"

// Note: rf_kernels/unit_testing.hpp is not included here!

#endif  // _RF_KERNELS_HPP
