#ifndef _RF_KERNELS_POLYNOMIAL_DETRENDER_HPP
#define _RF_KERNELS_POLYNOMIAL_DETRENDER_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct polynomial_detrender {
    const int axis;
    const int polydeg;
    
    // Function pointer to low-level kernel
    // Usage: _detrend(nfreq, nt, intensity, weights, stride, epsilon)    
    void (*_detrend)(int, int, float *, float *, int, double);

    // The constructor will automatically initialize the low-level kernel (_detrend).
    // The 'axis' argument should be 0 to fit along the frequency axis, or 1 to fit along the time axis.
    polynomial_detrender(int axis, int polydeg);

    // Note that the weights are not 'const' (in constrast to spline_detrender).
    // This is because the polynomial_detrender masks regions where the fit is poorly conditioned.
    void detrend(int nfreq, int nt, float *intensity, float *weights, int stride, double epsilon);
    
    // Function pointer to low-level kernel
    // Usage: _f(nfreq, nt, intensity, weights, stride, epsilon)    
    void (*_f)(int, int, float *, float *, int, double) = nullptr;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_POLYNOMIAL_DETRENDER_HPP
