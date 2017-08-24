#include <iostream>
#include "immintrin.h"

#include "rf_kernels/internals.hpp"
#include "rf_kernels/xorshift_plus.hpp"
#include "rf_kernels/online_mask_filler.hpp"

using namespace std;

namespace rf_kernels {
#if 0
};  // pacify emacs c-mode
#endif


inline void print_arr(__m256 a)
{
    // Helper function to print __m256 register (float[8])
    float arr[8];
    _mm256_storeu_ps((float *) &arr, a);
    for (int i=0; i<8; ++i)
        cout << arr[i] << " ";
    cout << "\n";
}


inline void print_arri(__m256i a)
{
    // Helper function to print __m256i register (int[8])
    int arr[8];
    _mm256_storeu_si256((__m256i *) &arr, a);
    for (int i=0; i<8; ++i)
        cout << arr[i] << " ";
    cout << "\n";
}


inline __m256 hadd(__m256 a)
{
    // Does a horizontal add of a __m256 register
    __m256 tmp0 = _mm256_add_ps(_mm256_permute2f128_ps(a, a, 0b00100001), a);
    __m256 tmp1 = _mm256_add_ps(_mm256_permute_ps(tmp0, 0b00111001), tmp0);
    return _mm256_add_ps(_mm256_permute_ps(tmp1, 0b01001110), tmp1);
}


inline void var_est(__m256 &v1, __m256 &wt1, __m256 w0, __m256 w1, __m256 w2, __m256 w3, __m256 i0, __m256 i1, __m256 i2, __m256 i3)
{
    // Does variance estimation for 32 intensity/weight values. Assumes mean=0.

    __m256 wi0 = _mm256_mul_ps(_mm256_mul_ps(i0, i0), w0);
    __m256 wi01 = _mm256_fmadd_ps(_mm256_mul_ps(i1, i1), w1, wi0);
    __m256 wi012 = _mm256_fmadd_ps(_mm256_mul_ps(i2, i2), w2, wi01);
    __m256 wi0123 = _mm256_fmadd_ps(_mm256_mul_ps(i3, i3), w3, wi012);
    __m256 vsum = hadd(wi0123);
    __m256 wsum = hadd(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(w0, w1), w2), w3));

    __m256 zero = _mm256_setzero_ps();
    __m256 mask = _mm256_cmp_ps(wsum, zero, _CMP_GT_OQ);
    __m256 wreg = _mm256_blendv_ps(_mm256_set1_ps(1.0), wsum, mask);

    v1 = _mm256_blendv_ps(zero, vsum/wreg, mask);
    wt1 = _mm256_set1_ps(0.03125) * wsum;
}


inline void update_var(__m256 v1, __m256 wt1, __m256 &prev_var, __m256 &prev_var_den, __m256 var_weight)
{
    __m256 a = prev_var_den - var_weight * prev_var_den;
    __m256 b = var_weight * wt1;
    __m256 c = a + b;

    __m256 mask = _mm256_cmp_ps(b, _mm256_setzero_ps(), _CMP_GT_OQ);
    __m256 creg = _mm256_blendv_ps(_mm256_set1_ps(1.0), c, mask);
    __m256 vreg = (a*prev_var + b*v1) / creg;

    prev_var = _mm256_blendv_ps(prev_var, vreg, mask);
    prev_var_den = c;
}


// Helper function which writes the first float32 in an __m256 to a given memory location 
// (Heuristically: *dst = src[0])  Surprisingly, this requires some hackery!
inline void _write_first(float *dst, __m256 src)
{
    __m128 src128 = _mm256_extractf128_ps(src, 0);
    int *dsti = reinterpret_cast<int *> (dst);  // _mm_extract_ps() returns int, not float?!
    *dsti = _mm_extract_ps(src128, 0);
}


template<bool multiply_intensity_by_weights, bool modify_weights>
void _online_mask_fill(online_mask_filler &params, int nt_chunk, int stride, float *intensity, float *weights)
{
    __m256 var_weight = _mm256_set1_ps(params.var_weight);
    __m256 w_clamp = _mm256_set1_ps(params.w_clamp);
    __m256 w_cutoff = _mm256_set1_ps(params.w_cutoff);

    int nfreq = params.nfreq;
    float *running_var = params.running_var.get();
    float *running_weights = params.running_weights.get();
    float *running_var_denom = params.running_var_denom.get();
    
    // Construct our random number generator object on the stack by initializing from the state kept in bonsai/online_mask_filler.c! 
    vec_xorshift_plus rng(params.rng_state);

    // Loop over frequencies first to avoid having to write the running_var and running_weights in each iteration of the ichunk loop
    for (int ifreq=0; ifreq<nfreq; ifreq++)
    {
      // Get the previous running_var and running_weights
      __m256 prev_w = _mm256_set1_ps(running_weights[ifreq]);
      __m256 prev_var = _mm256_set1_ps(running_var[ifreq]);
      __m256 prev_var_den = _mm256_set1_ps(running_var_denom[ifreq]);

      for (int ichunk=0; ichunk<nt_chunk-1; ichunk += 32)
      {
	  // Load intensity and weight arrays
	  __m256 i0 = _mm256_loadu_ps(intensity + ifreq * stride + ichunk);
	  __m256 i1 = _mm256_loadu_ps(intensity + ifreq * stride + ichunk + 8);
	  __m256 i2 = _mm256_loadu_ps(intensity + ifreq * stride + ichunk + 16);
	  __m256 i3 = _mm256_loadu_ps(intensity + ifreq * stride + ichunk + 24);

	  __m256 w0 = _mm256_loadu_ps(weights + ifreq * stride + ichunk);
	  __m256 w1 = _mm256_loadu_ps(weights + ifreq * stride + ichunk + 8);
	  __m256 w2 = _mm256_loadu_ps(weights + ifreq * stride + ichunk + 16);
	  __m256 w3 = _mm256_loadu_ps(weights + ifreq * stride + ichunk + 24);
	      
	  // Here, we do the variance computation:
	  __m256 v1, wt1;
	  var_est(v1, wt1, w0, w1, w2, w3, i0, i1, i2, i3);

	  // Then, use update rules to update running variance (prevent it from changing too much)
	  update_var(v1, wt1, prev_var, prev_var_den, var_weight);

	  // We also need to modify the weight values
	  wt1 = _mm256_max_ps(wt1, prev_w - w_clamp);
	  wt1 = _mm256_min_ps(wt1, prev_w + w_clamp);
	  prev_w = wt1;

	  __m256 root_three = _mm256_set1_ps(1.732050808);
	  __m256 rng_scale = root_three * _mm256_sqrt_ps(prev_var);

	  // Finally, mask fill with the running variance -- if weights less than cutoff, fill
	  i0 = _mm256_blendv_ps(i0, _mm256_mul_ps(rng.gen_floats(), rng_scale), _mm256_cmp_ps(w0, w_cutoff, _CMP_LE_OS));
	  i1 = _mm256_blendv_ps(i1, _mm256_mul_ps(rng.gen_floats(), rng_scale), _mm256_cmp_ps(w1, w_cutoff, _CMP_LE_OS));
	  i2 = _mm256_blendv_ps(i2, _mm256_mul_ps(rng.gen_floats(), rng_scale), _mm256_cmp_ps(w2, w_cutoff, _CMP_LE_OS));
	  i3 = _mm256_blendv_ps(i3, _mm256_mul_ps(rng.gen_floats(), rng_scale), _mm256_cmp_ps(w3, w_cutoff, _CMP_LE_OS));

	  if (multiply_intensity_by_weights) {
	      i0 *= prev_w;
	      i1 *= prev_w;
	      i2 *= prev_w;
	      i3 *= prev_w;
	  }
	  
	  // Store the new intensity values
	  _mm256_storeu_ps(intensity + ifreq * stride + ichunk, i0);
	  _mm256_storeu_ps(intensity + ifreq * stride + ichunk + 8, i1);
	  _mm256_storeu_ps(intensity + ifreq * stride + ichunk + 16, i2);
	  _mm256_storeu_ps(intensity + ifreq * stride + ichunk + 24, i3);

	  if (modify_weights) {
	      // Store the new weight values
	      _mm256_storeu_ps(weights + ifreq * stride + ichunk, prev_w);
	      _mm256_storeu_ps(weights + ifreq * stride + ichunk + 8, prev_w);
	      _mm256_storeu_ps(weights + ifreq * stride + ichunk + 16, prev_w);
	      _mm256_storeu_ps(weights + ifreq * stride + ichunk + 24, prev_w);
	  }
      }

      // Since we've now completed all the variance estimation and filling for this frequency channel 
      // in this chunk, we write our running variance and weight to the vector.

      _write_first(&running_var[ifreq], prev_var);
      _write_first(&running_weights[ifreq], prev_w);
      _write_first(&running_var_denom[ifreq], prev_var_den);
    }

    // Now that we're done, write out the new state of the random number generator back to the stack!
    rng.store_state(params.rng_state);
}


// -------------------------------------------------------------------------------------------------


online_mask_filler::online_mask_filler(int nfreq_) :
    nfreq(nfreq_)
{
    if (nfreq <= 0)
	throw runtime_error("rf_kernels::online_mask_filler: expected nfreq > 0");

    this->running_var = unique_ptr<float[]> (new float[nfreq]);
    this->running_weights = unique_ptr<float[]> (new float[nfreq]);
    this->running_var_denom = unique_ptr<float[]> (new float[nfreq]);

    memset(running_var.get(), 0, nfreq * sizeof(float));
    memset(running_weights.get(), 0, nfreq * sizeof(float));
    memset(running_var_denom.get(), 0, nfreq * sizeof(float));

    std::random_device rd;
    
    for (int i = 0; i < 8; i++)
	this->rng_state[i] = uint64_t(rd()) + (uint64_t(rd()) << 32);
}


// Helper for online_mask_filler::mask_fill()
inline void _check_args(const online_mask_filler &params, int nt_chunk, int stride, float *intensity, float *weights)
{
    if (nt_chunk <= 0)
	throw runtime_error("rf_kernels::online_mask_filler: expected nt_chunk > 0");
    if (params.nfreq <= 0)
	throw runtime_error("rf_kernels::online_mask_filler: expected nfreq > 0");
    if (params.v1_chunk != 32)
	throw runtime_error("rf_kernels::online_mask_filler: only v1_chunk=32 is currently implemented");
    if (nt_chunk % params.v1_chunk != 0)
	throw runtime_error("rf_kernels::online_mask_filler: expected nt_chunk to be a multiple of v1_chunk");
    if (abs(stride) < nt_chunk)
	throw runtime_error("rf_kernels::online_mask_filler: expected abs(stride) >= nt_chunk");
    if (params.var_weight <= 0.0)
	throw runtime_error("rf_kernels::online_mask_filler: var_weight is uninitialized (or <= 0)");
    if (params.w_clamp <= 0.0)
	throw runtime_error("rf_kernels::online_mask_filler: w_clamp is uninitialized (or <= 0)");
    if (params.w_cutoff < 0.0)
	throw runtime_error("rf_kernels::online_mask_filler: w_cutoff is uninitialized (or < 0)");
    if (intensity == nullptr)
	throw runtime_error("rf_kernels::online_mask_filler: 'intensity' pointer is null");
    if (weights == nullptr)
	throw runtime_error("rf_kernels::online_mask_filler: 'weights' pointer is null");
}


// There are now four versions of the online mask filler. If overwrite_on_wt0 is true, when the running weight drops to zero, 
// the running variance will be updated to the first successful variance estimate, instead of being restricted by the variance
// clamp parameters. If modify_weights is true, weights will be set to the running weights and the intensity will be set to the 
// sqrt(3 * running_var) * rng if the value was masked, or left alone if not. If modify_weights is false, the weights will not
// be modified and the intensity will be sqrt(3 * rv) * rw * rng if masked and intensity * rw if unmasked.

// modify_weights = false is required for bonsai
// modify_weights = true makes the most sense for rf_pipelines

void online_mask_filler::mask_fill(int nt_chunk, int stride, float *intensity, float *weights)
{
    _check_args(*this, nt_chunk, stride, intensity, weights);

    if (multiply_intensity_by_weights)
    {
        if (modify_weights)
  	    _online_mask_fill<true, true> (*this, nt_chunk, stride, intensity, weights);
        else
	    _online_mask_fill<true, false> (*this, nt_chunk, stride, intensity, weights);
    }
    else
    {
        if (modify_weights)
  	    _online_mask_fill<false, true> (*this, nt_chunk, stride, intensity, weights);
        else
	    _online_mask_fill<false, false> (*this, nt_chunk, stride, intensity, weights);
    }
}


// ----------------------------------------------------------------------------------------------------


void online_mask_filler::scalar_mask_fill(int nt_chunk, int stride, float *intensity, float *weights)
{
    _check_args(*this, nt_chunk, stride, intensity, weights);

    // This ordering of constructor arguments makes xorshift_plus (scalar code)
    // equivalent to vec_xorshift_plus (vector code).
    uint64_t *rs = this->rng_state;    
    xorshift_plus rng(rs[0], rs[4], rs[1], rs[5], rs[2], rs[6], rs[3], rs[7]);
    
    float rn[8]; // holds random numbers for mask filling
    
    // outer looop over frequency channels
    for (int ifreq = 0; ifreq < nfreq; ifreq++)
    {
	// (running_variance, running_weights, running_var_denom) for this frequency channel
	float rv = running_var[ifreq];
	float rw = running_weights[ifreq];
	float rvd = running_var_denom[ifreq];

	// middle loop over v1_chunks
	for (int ichunk = 0; ichunk < nt_chunk; ichunk += v1_chunk)
	{
	    // (intensity, weights) pointers for this v1_chunk
	    float *iacc = &intensity[ifreq*stride + ichunk];
	    float *wacc = &weights[ifreq*stride + ichunk];

	    float vsum = 0;
	    float wsum = 0;

	    for (int i = 0; i < v1_chunk; i++) {
		vsum += iacc[i] * iacc[i] * wacc[i];
		wsum += wacc[i];
	    }

	    // Variance, weight estimate from current v1 chunk.
	    float v1 = (wsum > 0.0) ? (vsum/wsum) : 0.0;
	    float w1 = wsum / v1_chunk;

	    // Update running_variance, running_variance_denom
	    float a = (1 - var_weight) * rvd;
	    float b = var_weight * w1;

	    rv = (a*rv + b*v1) / (a+b);
	    rvd = a+b;

	    // Update running_weights.
	    w1 = min(w1, rw + w_clamp);
	    w1 = max(w1, rw - w_clamp);
	    rw = w1;

	    for (int i = 0; i < v1_chunk; i++) {
		if (i % 8 == 0)
		    rng.gen_floats(rn);
		iacc[i] = (wacc[i] <= w_cutoff) ? (rn[i%8] * sqrt(3*rv)) : iacc[i];
	    }
	    
	    if (multiply_intensity_by_weights) {
		for (int i = 0; i < v1_chunk; i++)
		    iacc[i] *= rw;
	    }

	    if (modify_weights) {
		for (int i = 0; i < v1_chunk; i++)
		    wacc[i] = rw;
	    }
	}
	
	// Store running weights and var
	running_var[ifreq] = rv;
	running_weights[ifreq] = rw;
	running_var_denom[ifreq] = rvd;
    }

    rs[0] = rng.seeds[0];
    rs[1] = rng.seeds[2];
    rs[2] = rng.seeds[4];
    rs[3] = rng.seeds[6];
    rs[4] = rng.seeds[1];
    rs[5] = rng.seeds[3];
    rs[6] = rng.seeds[5];
    rs[7] = rng.seeds[7];
}


}  // namespace rf_kernels
