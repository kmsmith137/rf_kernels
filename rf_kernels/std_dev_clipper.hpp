#ifndef _RF_KERNELS_STD_DEV_CLIPPER_HPP
#define _RF_KERNELS_STD_DEV_CLIPPER_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

// enum axis_type is declared here
#include "core.hpp"

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct std_dev_clipper {
    const int nfreq;
    const int nt_chunk;

    const axis_type axis;
    const int Df;
    const int Dt;
    
    const double sigma;
    const bool two_pass;

    std_dev_clipper(int nfreq, int nt_chunk, axis_type axis, double sigma, int Df=1, int Dt=1, bool two_pass=true);
    ~std_dev_clipper();

    void clip(const float *intensity, float *weights, int stride);

    int nfreq_ds = 0;
    int nt_ds = 0;

    // FIXME overkill?
    float *tmp_i = nullptr;  // (nfreq_ds * nt_ds)
    float *tmp_w = nullptr;  // (nfreq_ds * nt_ds)
    float *tmp_v = nullptr;  // max(nfreq_ds, nt_ds)

    // Function pointer to low-level kernel.
    void (*_f)(std_dev_clipper *, const float *, float *, int) = nullptr;

    // Scalar helper called by kernel
    void _clip_1d();

    // Disallow copying, since we use bare pointers managed with malloc/free.
    std_dev_clipper(const std_dev_clipper &) = delete;
    std_dev_clipper& operator=(const std_dev_clipper &) = delete;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_STD_DEV_CLIPPER_HPP
