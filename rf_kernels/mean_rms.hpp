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
    const axis_type axis;
    const int Df;
    const int Dt;
    
    const int niter;
    const double sigma;
    const bool two_pass;

    weighted_mean_rms(int nfreq, int nt_chunk, axis_type axis=AXIS_NONE, int Df=1,
		      int Dt=1, int niter=1, double sigma=0, bool two_pass=true);

    ~weighted_mean_rms();
    
    void compute_wrms(const float *intensity, const float *weights, int stride);

    // Function pointer to low-level kernel.
    void (*_f)(const weighted_mean_rms *, const float *, const float *, int) = nullptr;

    int nfreq_ds = 0;
    int nt_ds = 0;
    int nout = 0;
    
    float *out_mean = nullptr;
    float *out_rms = nullptr;
    float *tmp_i = nullptr;  // shape (nfreq/Df, nt/Dt)
    float *tmp_w = nullptr;  // shape (nfreq/Df, nt/Dt)

    // Disallow copying, since we use bare pointers managed with malloc/free.
    weighted_mean_rms(const weighted_mean_rms &) = delete;
    weighted_mean_rms& operator=(const weighted_mean_rms &) = delete;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_MEAN_RMS_HPP
