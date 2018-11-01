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


class quantizer {
public:
    const int nbits;
    const int kernel_size;  // 256 on AVX machine, 512 on AVX-512
    
    // Currently, only nbits=1 is implemented!
    // In this case, the "quantizer" converts a floating-point input array to a bitmask,
    // where each output bit is 1 iff the corresponding input element is > 0.
    quantizer(int nbits);
    
    // Arguments:
    //   in: float32 array of shape (nfreq, nt)
    //   out: uint8_t array of shape (nfreq, (nt*nbits)/8).
    //   istride: stride of input array, in units sizeof(float), i.e. istride=nt for contiguous array
    //   ostride: stride of output array, in units sizeof(char), i.e. ostride=nt/8 for contiguous array
    // 
    // Caller must ensure that 'nt' is a multiple of 'kernel_size'.
    // Note: when nbits > 1 is implemented, quantize() will have more arguments.
    void quantize(int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride) const;

    // Slow reference kernel (for testing, or for reference when reading code).
    void slow_reference_quantize(int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride) const;
    
protected:
    // Function pointer to low-level kernel.
    void (*_f)(int, int, uint8_t *, int, const float *, int);
};


struct dequantizer {
    const int nbits;
    const int kernel_size;
    
    // Currently, only nbits=1 is implemented!
    dequantizer(int nbits);

    // Applies a boolean mask to an _existing_ floating-point array.
    // Input is a uint8_t array of shape (nfreq, nt/8).
    // Output is a float array of shape (nfreq, nt).
    // Caller must ensure that 'nt' is a multiple of 'kernel_size'.
    void apply_bitmask(int nfreq, int nt, float *out, int ostride, const uint8_t *in, int istride) const;
    
    // Function pointer to low-level kernel.
    void (*_f_bm)(int, int, float *, int, const uint8_t *, int);
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_QUANTIZE_HPP
