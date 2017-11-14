#include <memory>

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

    const int nfreq;

    // Tunable parameters.
    // Note that the defaults are generally invalid parameter values!
    // This is so we can throw an exception if the caller forgets to initialize them.
    int v1_chunk = 32;
    float var_weight = 0.0;         // running_variance decay constant per v1_chunk
    float w_clamp = 0.0;            // change in running_weight (either positive or negative) per v1_chunk
    float w_cutoff = -1.0;          // threshold weight below which intensity is considered masked

    // This is the fast kernel!
    // 'intensity' and 'weights' are 2D arrays of shape (nfreq, nt_chunk), with strides (istride, wstride).
    // Both arrays are "mask-filled" and modified in place.  (This version of the kernel is used in rf_pipelines.)
    void mask_fill_in_place(int nt_chunk, float *intensity, int istride, float *weights, int wstride);

    // This version of the kernel leaves the input 'intensity' and 'weights' arrays unmodified,
    // and instead writes the product (mask-filled intensity) * (mask-filled weights) to its output
    // array.  (This version of the kernel is used in bonsai.)
    void mask_fill_and_multiply(int nt_chunk, float *out, int ostride, const float *intensity, int istride, const float *weights, int wstride);
    
    // Slow reference version of mask_fill_in_place(), for testing.
    void scalar_mask_fill_in_place(int nt_chunk, float *intensity, int istride, float *weights, int wstride);

    // Persistent state, kept between calls to mask_fill().
    // Each of these is a 1D array of length nfreq.
    //
    // Note: chunk_min_weights and chunk_max_weights are an experiment that I
    // didn't end up using, but they don't slow down the code, so I left them in
    // for now!  I might remove them eventually...

    std::unique_ptr<float[]> running_var;
    std::unique_ptr<float[]> running_weights;
    std::unique_ptr<float[]> running_var_denom;
    std::unique_ptr<float[]> chunk_min_weight;   // mininum weight in last chunk processed.
    std::unique_ptr<float[]> chunk_max_weight;   // maximum weight in last chunk processed.
        
    uint64_t rng_state[8];
    
    // Disallow copying (since we use std::unique_ptr).
    online_mask_filler(const online_mask_filler &) = delete;
    online_mask_filler& operator=(const online_mask_filler &) = delete;
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_ONLINE_MASK_FILLER_HPP
