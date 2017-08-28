#ifndef _RF_KERNELS_DOWNSAMPLE_HPP
#define _RF_KERNELS_DOWNSAMPLE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct wi_downsampler {
    const int Df;
    const int Dt;

    wi_downsampler(int Df, int Dt);

    // Note: stride currently assumed to be the same for intensity, weights.
    void downsample(int nfreq_out, int nt_out, float *out_i, float *out_w,
		    int ostride, const float *in_i, const float *in_w, int istride);
    
    // Function pointer to low-level kernel
    void (*_f)(int, int, float *, float *, int, const float *, const float *, int, int, int);
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_DOWNSAMPLE_HPP
