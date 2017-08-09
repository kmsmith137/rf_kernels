#include "xorshift_plus.hpp"

#ifndef _RF_KERNELS_ONLINE_MASK_FILLER_HPP
#define _RF_KERNELS_ONLINE_MASK_FILLER_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct online_mask_filler_params {
    int v1_chunk = 32;
    float var_weight = 2.0e-3;      // running_variance decay constant
    float var_clamp_add = 1.0e-10;  // max allowed additive change in running_variance per v1_chunk (setting to a small value disables)
    float var_clamp_mult = 3.3e-3;  // max allowed fractional change in running_variance per v1_chunk
    float w_clamp = 3.3e-3;         // change in running_weight (either positive or negative) per v1_chunk
    float w_cutoff = 0.5;           // threshold weight below which intensity is considered masked
    bool overwrite_on_wt0 = true;
};


// 'intensity' and 'weights' are 2D arrays of shape (nfreq, nt_chunk) with spacing 'stride' between frequency channels.
// 'running_var' and 'running_weights' are 1D arrays of length nfreq.
extern void online_mask_fill(const online_mask_filler_params &params, int nfreq, int nt_chunk, int stride,
			     float *intensity, const float *weights, float *running_var, float *running_weights, 
			     uint64_t rng_state[8]);


extern void scalar_online_mask_fill(const online_mask_filler_params &params, int nfreq, int nt_chunk, int stride,
				    float *intensity, const float *weights, float *running_var, float *running_weights, 
				    xorshift_plus &rng);


}  // namespace rf_kernels

#endif  // _RF_KERNELS_ONLINE_MASK_FILLER_HPP
