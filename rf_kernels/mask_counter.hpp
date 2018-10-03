#ifndef _RF_KERNELS_MASK_COUNTER_HPP
#define _RF_KERNELS_MASK_COUNTER_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <cstdint>
#include "core.hpp"

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct mask_counter_data {
    int nfreq = 0;
    int nt_chunk = 0;

    const float *in = nullptr;   // 2d array of shape (nfreq, nt_chunk)
    int istride = 0;             // stride in input array (=nt_chunk for contiguous array)

    // If the out_bitmask and/or out_fcounts pointers are null, then
    // the bitmask/fcounts will not be computed by the kernel.

    uint8_t *out_bitmask = nullptr;   // 2d array of shape (nfreq, nt_chunk/8)
    int *out_fcounts = nullptr;       // 1d array of length nfreq
    int out_bmstride = 0;             // stride in bitmask array (=nt_chunk/8 for contiguous array)

    // The return value is the total number of unmasked samples.
    int mask_count() const;
    int slow_reference_mask_count() const;

    // Throws verbose exception on failure.
    void check_args() const;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_MASK_COUNTER_HPP
