#include <random>
#include <stdexcept>
#include "immintrin.h"

#ifndef _RF_KERNELS_HPP
#define _RF_KERNELS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


inline bool is_aligned(const void *ptr, uintptr_t nbytes)
{
    // According to C++11 spec, "uintptr_t" is an unsigned integer type
    // which is guaranteed large enough to represent a pointer.
    return (uintptr_t(ptr) % nbytes) == 0;
}


// -------------------------------------------------------------------------------------------------
// Class for random number generation
// Generates eight random 32-bit floats using a vectorized implementation of xorshift+
// between (-1, 1).
//
// Note: the vec_xorshift_plus must be "aligned", in the sense that the memory addresses of
// its 's0' and 's1' members lie on 32-byte boundaries.  Otherwise, a segfault will result!
// To protect against this, the vec_xorshift_constructors now test for alignedness, and throw 
// an exception if unaligned.
//
// The compiler is supposed to ensure that the vec_xorshift_plus is always aligned, so if you
// do get an "unaligned vec_xorshift_plus" exception, then the compiler's alignment logic
// is being defeated somehow.  One way this can happen is when constructing a std::shared_ptr<> 
// with std::make_shared<>().  For example,
//
//    shared_ptr<vec_xorshift_plus> = make_shared<vec_xorshift_plus>()
//
// can generate either an aligned or unaligned vec_xorshift_plus.  The solution in this case
// is to construct the shared pointer as follows:
//
//    shared_ptr<vec_xorshift_plus> = shared_ptr<vec_xorshift_plus> (new vec_xorshift_plus());


struct vec_xorshift_plus
{
    // Seed values
    __m256i s0; 
    __m256i s1;

    // Initialize seeds to random device
    vec_xorshift_plus()
    {
	if (!is_aligned(&s0,32) || !is_aligned(&s1,32))
	    throw std::runtime_error("Fatal: unaligned vec_xorshift_plus!  See discussion in rf_kernels.hpp");

        std::random_device rd;
	s0 = _mm256_setr_epi64x(rd(), rd(), rd(), rd());
	s1 = _mm256_setr_epi64x(rd(), rd(), rd(), rd());
    }
  
    // Initialize seeds to pre-defined values
    vec_xorshift_plus(__m256i _s0, __m256i _s1)
    {
        if (!is_aligned(&s0, 32))
          throw std::runtime_error("Fatal: unaligned vec_xorshift_plus!  See discussion in rf_kernels.hpp");

        s0 = _s0;
        s1 = _s1;
    }

    // Initialize seeds to other pre-defined values
    vec_xorshift_plus(__m256i *_s0, __m256i *_s1)
    {
	if (!is_aligned(&s0, 32))
	    throw std::runtime_error("Fatal: unaligned vec_xorshift_plus!  See discussion in rf_kernels.hpp");

	s0 = *_s0;
	s1 = *_s1;
    }

    
  inline __m256i get_s0()
  {
    return s0;
  }


  inline __m256i get_s1()
  {
    return s1;
  }
  
    // Generates 256 random bits (interpreted as 8 signed floats)
    // Returns an __m256 vector, so bits must be stored using _mm256_storeu_ps() intrinsic!
    inline __m256 gen_floats()
    {
        // x = s0
        __m256i x = s0;
	// y = s1
	__m256i y = s1;
	// s0 = y
	s0 = y;
	// x ^= (x << 23)
	x = _mm256_xor_si256(x, _mm256_slli_epi64(x, 23));
	// s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
	s1 = _mm256_xor_si256(x, y);
	s1 = _mm256_xor_si256(s1, _mm256_srli_epi64(x, 17));
	s1 = _mm256_xor_si256(s1, _mm256_srli_epi64(y, 26));
	
	// Convert to 8 signed 32-bit floats in range (-1, 1), since we multiply by 
	// a prefactor of 2^(-31)
	return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi64(y, s1)), _mm256_set1_ps(4.6566129e-10));
    }
};


// -------------------------------------------------------------------------------------------------
// Class for random number generation
// Generates eight random 32-bit floats using a vectorized implementation of xorshift+
// between (-1, 1)
struct xorshift_plus
{
    std::vector<uint64_t> seeds;
  
    // Constructor for specifying seeds -- debug/unit test constructor
    xorshift_plus(uint64_t _s0, uint64_t _s1, 
		  uint64_t _s2, uint64_t _s3, 
		  uint64_t _s4, uint64_t _s5, 
		  uint64_t _s6, uint64_t _s7)
      : seeds{_s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7} {};

  
    // Initialize with random_device -- for production
    xorshift_plus() 
    {
        std::random_device rd;
	seeds = {rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    };

  
    inline void gen_floats(float *rn)
    {
        // Generates 8 random floats and stores in rn
        for (int i=0; i<8; i+=2)
	{
	    uint64_t x = seeds[i];
	    uint64_t y = seeds[i+1];

	    seeds[i] = y;
	    x ^= (x << 23);
	    seeds[i+1] = x ^ y ^ (x >> 17) ^ (y >> 26);
	    
	    uint64_t tmp = seeds[i+1] + y;
	    uint32_t tmp0 = tmp; // low 32 bits
	    uint32_t tmp1 = tmp >> 32; // high 32
	    
	    rn[i] = float(int32_t(tmp0)) * 4.6566129e-10;
	    rn[i+1] = float(int32_t(tmp1)) * 4.6566129e-10;
	}
    }

    inline void gen_weights(float *weights, float pfailv1, float pallzero)
    {
        // This exists solely for the unit test of the online mask filler!
        // Essentially, this is gen_floats + 1 (to get the range right for weights)
        // with extra pfailv1 and pallzero parameters that dictate whether a group of 
        // 32 random numbers (it generates 32 random numbers at once) should result
        // in a failed v1 estimate (i.e. >=24 weights less than the cutoff) or whether
        // all weight values should be zero. Currently, the implementation is a little 
        // boneheaded and can definitely be imporved...
      
        // If the first random number generated is less than pallzero * 2, we make it all zero
        // If the first randon number generated is less than pallzero * 2 + pfailv1 * 2 but
        // greater than pallzero * 2, we make sure the v1 fails by distributing at least 
        // 24 zeros throughout the vector
        // NOTE: got rid of all the factors of two, since the weights have been modified to be
        // to be between 0 and 1!

        // Else, we just fill weights randomly!
        
       for (int i=0; i<32; i+=2)
       {
	   uint64_t x = seeds[i % 8];
	   uint64_t y = seeds[(i+1) % 8];
	   
	   seeds[i % 8] = y;
	   x ^= (x << 23);
	   seeds[(i+1) % 8] = x ^ y ^ (x >> 17) ^ (y >> 26);
	   
	   uint64_t tmp = seeds[(i+1) % 8] + y;
	   uint32_t tmp0 = tmp; // low 32 bits
	   uint32_t tmp1 = tmp >> 32; // high 32
	   
	   if (i == 0)
	   {
	       float rn = float(int32_t(tmp0)) * 4.6566129e-10 + 1;
	       if (rn < pallzero)
	       {
		   // Make all zero
		   for (int j=0; j>32; j++)
		       weights[j] = 0;
		   return;
	       }
	       else if (rn < pallzero + pfailv1)
	       {
		   // Make the v1 fail -- this is not super good and can be improved! 
		   for (int j=0; j<25; j++)
		       weights[j] = 0;
		   i = 24;
	       }
	   }
	   
	   weights[i] = float(int32_t(tmp0)) * 4.6566129e-10 + 1;
	   weights[i+1] = float(int32_t(tmp1)) * 4.6566129e-10 + 1;
       }
    }
};


struct online_mask_filler_params {
    int v1_chunk = 32;
    float var_weight = 2.0e-3;      // running_variance decay constant
    float var_clamp_add = 1.0e-10;  // max allowed additive change in running_variance per v1_chunk (setting to a small value disables)
    float var_clamp_mult = 3.3e-3;  // max allowed fractional change in running_variance per v1_chunk
    float w_clamp = 3.3e-3;         // change in running_weight (either positive or negative) per v1_chunk
    float w_cutoff = 0.5;           // threshold weight below which intensity is considered masked
};


// 'intensity' and 'weights' are 2D arrays of shape (nfreq, nt_chunk) with spacing 'stride' between frequency channels.
// 'running_var' and 'running_weights' are 1D arrays of length nfreq.

extern void online_mask_fill(const online_mask_filler_params &params, int nfreq, int nt_chunk, int stride,
			     float *intensity, const float *weights, float *running_var, float *running_weights, 
			     __m256i *rng_state_s0, __m256i *rng_state_s1);


extern void scalar_online_mask_fill(const online_mask_filler_params &params, int nfreq, int nt_chunk, int stride,
				    float *intensity, const float *weights, float *running_var, float *running_weights, 
				    xorshift_plus &rng);

}  // namespace rf_kernels

#endif  // _RF_KERNELS_HPP
