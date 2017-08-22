// Currently, can only spline-detrend in frequency direction!

#ifndef _RF_KERNELS_SPLINE_DETRENDER_HPP
#define _RF_KERNELS_SPLINE_DETRENDER_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++0x support (g++ -std=c++0x)"
#endif

namespace rf_kernels {
#if 0
}  // emacs pacifier
#endif


// 'bin_delim' is a 1D array of length (nbins+1).
// 'poly_vals' is a 2D array of shape (nx,4).
extern void _spline_detrender_init(int *bin_delim, float *poly_vals, int nx, int nbins);


}  // namespace rf_kernels

#endif  // _RF_KERNELS_SPLINE_DETRENDER_HPP
