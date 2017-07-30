#include "rf_kernels.hpp"
#include "immintrin.h"
#include <iostream>

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


inline __m256 check_weights(__m256 w0, __m256 w1, __m256 w2, __m256 w3)
{
    // Check the number of weights that are non-zero
    // Returns a constant register containing the number of _successful_ weights
    __m256 zero = _mm256_set1_ps(0.0);

    // Note here that _mm256_cmp_ps returns a _m256 in which groups of 8 bits are either all 1s if the cmp was true and all 0s if the cmp was false
    // If we reinterpret this as an _m256i, we get a register in which all values are either 0 (in the case of 8 zeros) or -1 (in the case of 8 ones
    // given that 11111111 is -1 in twos complement form). Thus, by doing an hadd at the end, we will get -1 * the number of successful intensities
    // Note that _mm256_castps_si256 just reinterprets bits whereas _mm256_cvtepi32_ps truncates a float into an int. 
    __m256i w0_mask = _mm256_castps_si256(_mm256_cmp_ps(w0, zero, _CMP_GT_OS));
    __m256i w1_mask = _mm256_castps_si256(_mm256_cmp_ps(w1, zero, _CMP_GT_OS));
    __m256i w2_mask = _mm256_castps_si256(_mm256_cmp_ps(w2, zero, _CMP_GT_OS));
    __m256i w3_mask = _mm256_castps_si256(_mm256_cmp_ps(w3, zero, _CMP_GT_OS));
    return hadd(_mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(w0_mask, w1_mask), w2_mask), w3_mask)));
}


inline __m256 var_est(__m256 w0, __m256 w1, __m256 w2, __m256 w3, __m256 i0, __m256 i1, __m256 i2, __m256 i3)
{
    // Does variance estimation for 32 intensity/weight values. Assumes mean=0.
    __m256 wi0 = _mm256_mul_ps(_mm256_mul_ps(i0, i0), w0);
    __m256 wi01 = _mm256_fmadd_ps(_mm256_mul_ps(i1, i1), w1, wi0);
    __m256 wi012 = _mm256_fmadd_ps(_mm256_mul_ps(i2, i2), w2, wi01);
    __m256 wi0123 = _mm256_fmadd_ps(_mm256_mul_ps(i3, i3), w3, wi012);
    __m256 vsum = hadd(wi0123);
    __m256 wsum = hadd(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(w0, w1), w2), w3));
    wsum = _mm256_max_ps(wsum, _mm256_set1_ps(1.0));
    return _mm256_div_ps(vsum, wsum);
}


inline __m256 update_var(__m256 tmp_var, __m256 prev_var, float var_weight, float var_clamp_add, float var_clamp_mult, __m256 mask)
{
    // Does the update of the running variance (tmp_var) by checking the exponential update (normal_upd) doesn't exceed the bounds (high/low) specified 
    // by var_clamp_add and var_clamp_mult
    __m256 normal_upd = _mm256_fmadd_ps(_mm256_set1_ps(var_weight), tmp_var, _mm256_mul_ps(_mm256_set1_ps(1 - var_weight), prev_var)); 
    __m256 high = _mm256_add_ps(_mm256_add_ps(prev_var, _mm256_set1_ps(var_clamp_add)), _mm256_mul_ps(prev_var, _mm256_set1_ps(var_clamp_mult))); 
    __m256 low = _mm256_sub_ps(_mm256_sub_ps(prev_var, _mm256_set1_ps(var_clamp_add)), _mm256_mul_ps(prev_var, _mm256_set1_ps(var_clamp_mult)));     
    __m256 ideal_update = _mm256_max_ps(_mm256_min_ps(normal_upd, high), low); 
    return _mm256_blendv_ps(prev_var, ideal_update, mask);
}



void online_mask_fill(const online_mask_filler_params &params, int nfreq, int nt_chunk, int stride,
			   float *intensity, const float *weights, float *running_var, float *running_weights,
			   vec_xorshift_plus &rng)
{
    const float w_clamp = params.w_clamp;
    const float var_weight = params.var_weight;
    const float var_clamp_add = params.var_clamp_add;
    const float var_clamp_mult = params.var_clamp_mult;
    const float w_cutoff = params.w_cutoff;

    __m256 tmp_var, prev_var, prev_w, w0, w1, w2, w3, i0, i1, i2, i3, res0, res1, res2, res3;
    __m256 c = _mm256_set1_ps(w_cutoff);
    __m256 root_three = _mm256_sqrt_ps(_mm256_set1_ps(3));
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 zero = _mm256_set1_ps(0.0f);


    // Loop over frequencies first to avoid having to write the running_var and running_weights in each iteration of the ichunk loop
    for (int ifreq=0; ifreq<nfreq; ifreq++)
    {
      // Get the previous running_var and running_weights
      prev_var = _mm256_set1_ps(running_var[ifreq]);
      prev_w = _mm256_set1_ps(running_weights[ifreq]);

      for (int ichunk=0; ichunk<nt_chunk-1; ichunk += 32)
	{
	  // Load intensity and weight arrays
	  i0 = _mm256_loadu_ps(intensity + ifreq * stride + ichunk);
	  i1 = _mm256_loadu_ps(intensity + ifreq * stride + ichunk + 8);
	  i2 = _mm256_loadu_ps(intensity + ifreq * stride + ichunk + 16);
	  i3 = _mm256_loadu_ps(intensity + ifreq * stride + ichunk + 24);
	  w0 = _mm256_loadu_ps(weights + ifreq * stride + ichunk);
	  w1 = _mm256_loadu_ps(weights + ifreq * stride + ichunk + 8);
	  w2 = _mm256_loadu_ps(weights + ifreq * stride + ichunk + 16);
	  w3 = _mm256_loadu_ps(weights + ifreq * stride + ichunk + 24);

	  // First, we need to see how many of the weights are greater than zero. Note that npass contains 
	  // a number equal to -1 * the number of successful intensities (see check_weights comments)
	  __m256 npass = check_weights(w0, w1, w2, w3);

	  // If pass is less than -8, we treat this as a failed v1 case (since >75% of the data is masked, we can't use that to update 
	  // our running variance estimate). After doing the compare below, mask will contain we get a constant register that is either 
	  // all zeros if the variance estimate failed or all ones if the variance estimate was successful
	  __m256 mask = _mm256_cmp_ps(npass, _mm256_set1_ps(-8.1), _CMP_LT_OS);

	  // If the running weight is 0, we want to set the running variance to the next successful v1 estimate and the running weight
	  // to w_clamp. To do this, make this mask rw_check that is all ones if prev_w is <= 0.0f.
	  __m256 rw_check = _mm256_cmp_ps(prev_w, zero, _CMP_LE_OS);
	      
	  // Here, we do the variance computation:
	  tmp_var = var_est(w0, w1, w2, w3, i0, i1, i2, i3);

	  // Then, use update rules to update value we'll eventually set as our running variance (prevent it from changing too much)
	  prev_var = update_var(tmp_var, prev_var, var_weight, var_clamp_add, var_clamp_mult, mask);

	  // We also need to modify the weight values
	  __m256 w = _mm256_blendv_ps(_mm256_set1_ps(-w_clamp), _mm256_set1_ps(w_clamp), mask);    // either +w_clamp or -w_clamp
	  w = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(w, prev_w), zero), one);

	  // Update the running variance and running weights based on rw_check
	  prev_var = _mm256_blendv_ps(prev_var, tmp_var, rw_check);
	  prev_w = _mm256_blendv_ps(w, _mm256_set1_ps(w_clamp), rw_check);

	  // Finally, mask fill with the running variance -- if weights less than cutoff, fill
	  res0 = _mm256_blendv_ps(_mm256_mul_ps(prev_w, i0), 
				  _mm256_mul_ps(prev_w, _mm256_mul_ps(rng.gen_floats(), _mm256_mul_ps(root_three, _mm256_sqrt_ps(prev_var)))), 
				  _mm256_cmp_ps(w0, c, _CMP_LT_OS));
	  res1 = _mm256_blendv_ps(_mm256_mul_ps(prev_w, i1),
				  _mm256_mul_ps(prev_w, _mm256_mul_ps(rng.gen_floats(), _mm256_mul_ps(root_three, _mm256_sqrt_ps(prev_var)))), 
				  _mm256_cmp_ps(w1, c, _CMP_LT_OS));
	  res2 = _mm256_blendv_ps(_mm256_mul_ps(prev_w, i2), 
				  _mm256_mul_ps(prev_w, _mm256_mul_ps(rng.gen_floats(), _mm256_mul_ps(root_three, _mm256_sqrt_ps(prev_var)))),
				  _mm256_cmp_ps(w2, c, _CMP_LT_OS));
	  res3 = _mm256_blendv_ps(_mm256_mul_ps(prev_w, i3), 
				  _mm256_mul_ps(prev_w, _mm256_mul_ps(rng.gen_floats(), _mm256_mul_ps(root_three, _mm256_sqrt_ps(prev_var)))), 
				  _mm256_cmp_ps(w3, c, _CMP_LT_OS));
	      
	  // Store the new intensity values - note that we no longer update or store the weights!
	  _mm256_storeu_ps((float*) (intensity + ifreq * stride + ichunk), res0);
	  _mm256_storeu_ps((float*) (intensity + ifreq * stride + ichunk + 8), res1);
	  _mm256_storeu_ps((float*) (intensity + ifreq * stride + ichunk + 16), res2);
	  _mm256_storeu_ps((float*) (intensity + ifreq * stride + ichunk + 24), res3);
	}
      // Since we've now completed all the variance estimation and filling for this frequency channel in this chunk, we must write our 
      // running variance and weight to the vector, which is a bit of a pain. Thanks for this hack, Kendrick!

      // First step: extract elements 0-3 into a 128-bit register.
      __m128 y = _mm256_extractf128_ps(prev_var, 0);
      __m128 z = _mm256_extractf128_ps(prev_w, 0);

      // The intrinsic _mm_extract_ps() extracts element 0 from the 128-bit register, but it has the wrong
      // return type (int32 instead of float32). The returned value is a "fake" int32 obtained by interpreting 
      // the bit pattern of the "real" float32 (in IEEE-754 representation) as an int32 (in twos-complement representation),
      // so it's not very useful. Nevertheless if we just write it to memory as an int32, and read back from 
      // the same memory location later as a float32, we'll get the right answer.
      int *i = reinterpret_cast<int *> (&running_var[ifreq]);   // hack: (int *) pointing to same memory location as q[0]
      int *j = reinterpret_cast<int *> (&running_weights[ifreq]);   
      *i = _mm_extract_ps(y, 0); // write a "fake" int32 to this memory location
      *j = _mm_extract_ps(z, 0);
    }

}


// ----------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------
// Based on Maya code in rf_pipelines (scalar_mask_fill() in online_mask_filler.cpp)
// with a few changes as follows which were helpful when incorporating it into bonsai.
//
//   - We don't actually modify the weights array (we do update the running_weights)
//     Not sure what's best here, maybe define a boolean flag for maximum flexibility?
//
//   - We do modify the intensity array (in the case where the intensity is unmasked),
//     by applying (i.e. multiplying by) the running_weight.
//
//   - For consistency, in the case where an intensity value is masked, we apply the
//     running_weight to the random value which is simulated.  I.e., the simulated
//     random intensity has variance (3 * running_weight^2 * running_variance).
//
//   - See "KMS" below for two more minor changes...

inline bool get_v1(const float *intensity, const float *weights, float &v1)
{
    int zerocount = 0;
    float vsum = 0;
    float wsum = 0;

    for (int i=0; i < 32; ++i)
    {
        // I assume this is okay for checking whether the weight is 0?
        if (weights[i] < 1e-7)
	    ++zerocount;
	vsum += intensity[i] * intensity[i] * weights[i];
	wsum += weights[i];
    }

    wsum = max(wsum, 1.0f);
    
    // Check whether enough valid values were passed
    if (zerocount >= 23.9)
    {
        v1 = 0;
	return false;
    }
    v1 = vsum / wsum;
    return true;
}

void scalar_online_mask_fill(const online_mask_filler_params &params, int nfreq, int nt_chunk, int stride,
		      float *intensity, const float *weights, float *running_var, float *running_weights,
		      xorshift_plus &rng)
{
    const int v1_chunk = params.v1_chunk;
    const float w_clamp = params.w_clamp;
    const float var_weight = params.var_weight;
    const float var_clamp_add = params.var_clamp_add;
    const float var_clamp_mult = params.var_clamp_mult;
    const float w_cutoff = params.w_cutoff;

    float rn[8]; // holds random numbers for mask filling
    
    // outer looop over frequency channels
    for (int ifreq = 0; ifreq < nfreq; ifreq++)
    {
        // (running_variance, running_weights) for this frequency channel
        float rv = running_var[ifreq];
	float rw = running_weights[ifreq];
	float v1;
	
	// middle loop over v1_chunks
	for (int ichunk = 0; ichunk < nt_chunk; ichunk += v1_chunk)
	{
	    // (intensity, weights) pointers for this v1_chunk
	    float *iacc = &intensity[ifreq*stride + ichunk];
	    const float *wacc = &weights[ifreq*stride + ichunk];
	    
	    if (!get_v1(iacc, wacc, v1))
	    {
	        // For an unsuccessful v1, we decrease the weight if possible. We do not modify the running variance
	        rw = max(0.0f, rw - w_clamp);
	    }
	    else if (rw <= 0.0f) 
	    {
	        // KMS: changed logic slightly here!  If the v1 is successful but the running_weight is zero,
	        // we set the running_variance equal to the value of 'v1' (ignoring the old value of running_variance),
	        // rather than slowly transitioning from the old variance estimate to the new one.
	        rv = v1;
		rw = w_clamp;
	    } 
	    else 
	    {
	        // If the v1 was succesful, try to increase the weight, if possible
	        // KMS: changed max weight from 2.0f to 1.0f here, since taking max_weight=2 was confusing one of my
	        // bonsai unit tests, and max_weight=2 was just catering to a bug of mine in rf_pipelines anyway!
	        rw = min(1.0f, rw + w_clamp);

		// Then, restrict the change in variance estimate defined by the clamp parameters
		v1 = (1 - var_weight) * rv + var_weight * v1;
		v1 = min(v1, rv + var_clamp_add + rv*var_clamp_mult);
		v1 = max(v1, rv - var_clamp_add - rv*var_clamp_mult);
		rv = v1;
	    }
	    
	    // Scaling factor for random numbers.
	    // KMS note factor of the running_weight here!
	    float scale = rw * sqrt(3*rv);
	    
	    for (int i = 0; i < v1_chunk; i++)
	    {
	        if (i % 8 == 0)
	  	rng.gen_floats(rn);
		iacc[i] = (wacc[i] < w_cutoff) ? rw * iacc[i] : rn[i % 8] * scale;
	    }
	}

	running_var[ifreq] = rv;
	running_weights[ifreq] = rw;
    }
}

}  // namespace rf_kernels
