#ifndef _RF_KERNELS_MEAN_RMS_HPP
#define _RF_KERNELS_MEAN_RMS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

// enum axis_type is declared here
#include "core.hpp"

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// I'll probably extend this later to support (axis, Df, Dt)!

struct weighted_mean_rms {
    const int nfreq;
    const int nt_chunk;
    
    const int niter;
    const double sigma;
    const bool two_pass;

    weighted_mean_rms(int nfreq, int nt_chunk, int niter=1, double sigma=0, bool two_pass=true);

    void compute_wrms(float &mean, float &rms, const float *intensity, const float *weights, int stride);
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_MEAN_RMS_HPP
