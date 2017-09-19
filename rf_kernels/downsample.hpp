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

    void downsample(int nfreq_out, int nt_out,
		    float *out_i, int out_istride,
		    float *out_w, int out_wstride,
		    const float *in_i, int in_istride,
		    const float *in_w, int in_wstride);
    
    // Function pointer to low-level kernel.
    void (*_f)(const wi_downsampler *, int, int, float *, int, float *, int, const float *, int, const float *, int);
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_DOWNSAMPLE_HPP
