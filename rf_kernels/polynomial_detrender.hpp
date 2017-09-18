#ifndef _RF_KERNELS_POLYNOMIAL_DETRENDER_HPP
#define _RF_KERNELS_POLYNOMIAL_DETRENDER_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

// enum axis_type is declared here
#include "core.hpp"

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct polynomial_detrender {
    const axis_type axis;
    const int polydeg;
    
    // The constructor will automatically initialize the low-level kernel (_detrend).
    // The 'axis' argument should either be AXIS_FREQ or AXIS_TIME.
    // (Note: 'enum axis_type' is defined in rf_kernels/core.hpp)
    polynomial_detrender(axis_type axis, int polydeg);

    // Note that the weights are not 'const' (in constrast to spline_detrender).
    // This is because the polynomial_detrender masks regions where the fit is poorly conditioned.
    void detrend(int nfreq, int nt, float *intensity, int istride, float *weights, int wstride, double epsilon);
    
    // Function pointer to low-level kernel
    // Usage: _f(nfreq, nt, intensity, istride, weights, wstride, epsilon)    
    void (*_f)(int, int, float *, int, float *, int, double) = nullptr;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_POLYNOMIAL_DETRENDER_HPP
