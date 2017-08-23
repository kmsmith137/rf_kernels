#include <memory>
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


struct online_mask_filler {
    // Constructing an online_mask_filler is a two-step process: first call 
    // the constructor, then set values of the "tunable" parameters below.

    explicit online_mask_filler(int nfreq);

    // Tunable parameters.
    // Note that the defaults are generally invalid parameter values!
    // This is so we can throw an exception if the caller forgets to initialize them.
    int v1_chunk = 32;
    float var_weight = 0.0;         // running_variance decay constant
    float var_clamp_add = 0.0;      // max allowed additive change in running_variance per v1_chunk (setting to a small value disables)
    float var_clamp_mult = 0.0;     // max allowed fractional change in running_variance per v1_chunk
    float w_clamp = 0.0;            // change in running_weight (either positive or negative) per v1_chunk
    float w_cutoff = -1.0;          // threshold weight below which intensity is considered masked
    bool overwrite_on_wt0 = true;   // if the weights drop to zero, overwrite the intensity with the first successful v1 instead of restricting it to v1_clamp
    bool modify_weights = false;    // if false, the weights are unmodified by the kernel, but the intensity is multiplied by the weights (wanted in bonsai)

    // This is the fast kernel!
    // 'intensity' and 'weights' are 2D arrays of shape (nfreq, nt_chunk) with spacing 'stride' between frequency channels.
    void mask_fill(int nt_chunk, int stride, float *intensity, float *weights);
    
    // Slow reference version of mask_fill(), for testing.
    void scalar_mask_fill(int nt_chunk, int stride, float *intensity, float *weights);

    // Persistent state, kept between calls to mask_fill().
    // 'running_var' and 'running_weights' are 1D arrays of length nfreq.
    const int nfreq;
    std::unique_ptr<float[]> running_var;
    std::unique_ptr<float[]> running_weights;
    uint64_t rng_state[8];
    
    // Disallow copying (since we use std::unique_ptr).
    online_mask_filler(const online_mask_filler &) = delete;
    online_mask_filler& operator=(const online_mask_filler &) = delete;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_ONLINE_MASK_FILLER_HPP
