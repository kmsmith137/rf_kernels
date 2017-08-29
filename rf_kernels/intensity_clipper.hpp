#ifndef _RF_KERNELS_INTENSITY_CLIPPER_HPP
#define _RF_KERNELS_INTENSITY_CLIPPER_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

// enum axis_type is declared here
#include "core.hpp"

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct intensity_clipper {
    const int nfreq;
    const int nt_chunk;
    const axis_type axis;
    const double sigma;
    
    const int Df;
    const int Dt;
    const int niter;
    const double iter_sigma;
    const bool two_pass;

    // Note: if 'iter_sigma' is zero, then 'sigma' will be used.
    intensity_clipper(int nfreq, int nt_chunk, axis_type axis, double sigma,
		      int Df=1, int Dt=1, int niter=1, double iter_sigma=0,
		      bool two_pass=false);
    
    ~intensity_clipper();

    void clip(const float *intensity, float *weights, int stride);

    float *ds_intensity = nullptr;
    float *ds_weights = nullptr;

    // Function pointer to low-level kernel.
    void (*_f)(const float *, float *, int, int, int, int, double, double, float *, float *) = nullptr;

    // Disallow copying, since we use bare pointers managed with malloc/free.
    intensity_clipper(const intensity_clipper &) = delete;
    intensity_clipper& operator=(const intensity_clipper &) = delete;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_INTENSITY_CLIPPER_HPP
