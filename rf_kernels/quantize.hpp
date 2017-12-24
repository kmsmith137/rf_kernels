#ifndef _RF_KERNELS_QUANTIZE_HPP
#define _RF_KERNELS_QUANTIZE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <cstdint>

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct quantizer {
    const int nbits;
    const int kernel_size;
    
    // Currently, only nbits=1 is implemented!
    quantizer(int nbits);
    
    // Output is a uint8_t array of shape (nfreq, (nt*nbits)/8).
    // Caller must ensure that 'nt' is a multiple of 'kernel_size'.
    // When nbits > 1 is implemented, this will have at least one more argument (scales).
    void quantize(int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride) const;
    
    // Function pointer to low-level kernel.
    void (*_f)(int, int, uint8_t *, int, const float *, int);
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_QUANTIZE_HPP
