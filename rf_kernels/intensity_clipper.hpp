#ifndef _RF_KERNELS_INTENSITY_CLIPPER_HPP
#define _RF_KERNELS_INTENSITY_CLIPPER_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct intensity_clipper {
    const int nfreq;
    const int nt_chunk;
    const axis_type axis;
    const int Df;
    const int Dt;
    const bool two_pass;

    intensity_clipper(int nfreq, int nt_chunk, axis_type axis, int Df, int Dt, bool two_pass);
    
    void clip(const float *intensity, float *weights, int stride, double sigma, int niter, double iter_sigma);

    // Function pointer to low-level kernel.
    void (*_f)(const float *, float *, int, int, int, int, double, double, float *, float *) = nullptr;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_INTENSITY_CLIPPER_HPP
