#ifndef _RF_KERNELS_UPSAMPLE_HPP
#define _RF_KERNELS_UPSAMPLE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct weight_upsampler {
    const int Df;
    const int Dt;

    weight_upsampler(int Df, int Dt);

    void upsample(int nfreq_in, int nt_in, float *out, int ostride, const float *in, int istride, float w_cutoff=0.0);
    
    // Function pointer to low-level kernel
    void (*_f)(int, int, float *, int, const float *, int, float, int, int) = nullptr;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_UPSAMPLE_HPP
