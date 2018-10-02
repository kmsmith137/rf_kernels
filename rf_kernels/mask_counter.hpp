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


class mask_counter {
public:
    struct initializer {
	int nfreq = 0;
	int nt_chunk = 0;
	bool save_bitmask = false;   // save 2d array indexed by (frequency, time)?
	bool save_tcounts = false;   // save 1d array indexed by time?
	bool save_fcounts = false;   // save 1d array indexed by frequency?
    };

    struct kernel_args {
	const float *in = nullptr;   // 2d array of shape (nfreq, nt_chunk)
	int istride = 0;             // stride in input array (=nt_chunk for contiguous array)

	uint8_t *out_bitmask = nullptr;   // 2d array of shape (nfreq, nt_chunk/8)
	int *out_tcounts = nullptr;       // 1d array of length nt_chunk
	int *out_fcounts = nullptr;       // 1d array of length nfreq
	int out_bmstride = 0;             // stride in bitmask array (=nt_chunk/8 for contiguous array)
    };

    const initializer ini_params;

    mask_counter(const initializer &ini_params);
    
    void mask_count(const kernel_args &args) const;

    void slow_reference_mask_count(const kernel_args &args) const;

    std::shared_ptr<uint8_t> _bm_workspace;

protected:
    void (*f_first_pass)(const mask_counter &, const kernel_args &) = nullptr;
    void (*f_second_pass)(const mask_counter &, const kernel_args &) = nullptr;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_MASK_COUNTER_HPP
