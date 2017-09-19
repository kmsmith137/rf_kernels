// FIXME: currently, can only spline-detrend in frequency direction!

#include <cstdint>

#ifndef _RF_KERNELS_SPLINE_DETRENDER_HPP
#define _RF_KERNELS_SPLINE_DETRENDER_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <cstdint>

namespace rf_kernels {
#if 0
}  // emacs pacifier
#endif


struct spline_detrender {
    // Disallow copying
    spline_detrender(const spline_detrender &) = delete;
    spline_detrender& operator=(const spline_detrender &) = delete;
    
    const int nfreq;
    const int nbins;
    const float epsilon;

    spline_detrender(int nfreq, int nbins, float epsilon=3.0e-4);
    ~spline_detrender();

    void detrend(int nt_chunk, float *intensity, int istride, const float *weights, int wstride);

    int *bin_delim = nullptr;           // length (nbins+1)
    float *poly_vals = nullptr;         // length (nfreq * 4)
    float *ninv = nullptr;              // length (nbins * 10 * S), where S is the simd size
    float *ninvx = nullptr;             // length (nbins * 4 * S)
    float *cholesky_invdiag = nullptr;  // length (nbins+1) * 3 * S
    float *cholesky_subdiag = nullptr;  // length (nbins * 4 * S)
    float *coeffs = nullptr;            // length (nbins+1) * 2 * S

    uint8_t *allocated_memory = nullptr;

    // Defined in rf_pipelines/spline_detrender_internal.hpp
    inline void _kernel_ninv(const float *intensity, int istride, const float *weights, int wstride);
    inline void _kernel_detrend(float *intensity, int istride);
    inline void _kernel_fit_pass1();
    inline void _kernel_fit_pass2();
    inline void _kernel_fit_pass3();
};


// Helper function called by spline_detrender constructor.
//  'bin_delim' is a 1D array of length (nbins+1).
//  'poly_vals' is a 2D array of shape (nx,4).
extern void _spline_detrender_init(int *bin_delim, float *poly_vals, int nx, int nbins);


}  // namespace rf_kernels

#endif  // _RF_KERNELS_SPLINE_DETRENDER_HPP
