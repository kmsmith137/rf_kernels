#include <random>
#include <stdexcept>
#include "immintrin.h"

#ifndef _RF_KERNELS_XORSHIFT_PLUS_HPP
#define _RF_KERNELS_XORSHIFT_PLUS_HPP

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
// Generally speaking, the compiler will correctly align the vec_xorshift_plus if it is allocated
// on the call stack of a function, but it may not be aligned if it is allocated in the heap (or
// embedded in a larger heap-allocated class).  In this case, one solution is to represent the
// persistent rng state as a uint64_t[8] (which can be in the heap).  When a vec_xorshift_plus is
// needed, a temporary one can be constructed on the stack, using load/store functions (see below)
// to exchange state with the uint64_t[8].


struct vec_xorshift_plus
{
    // Seed values
    __m256i s0; 
    __m256i s1;

    // Initialize seeds to random device
    vec_xorshift_plus()
    {
	if (!is_aligned(&s0, 32) || !is_aligned(&s1, 32))
	    throw std::runtime_error("Fatal: unaligned vec_xorshift_plus!  See discussion in rf_kernels.hpp");

        std::random_device rd;
	s0 = _mm256_setr_epi64x(rd(), rd(), rd(), rd());
	s1 = _mm256_setr_epi64x(rd(), rd(), rd(), rd());
    }
  
    // Initialize seeds to pre-defined values (dangerous if __m256is are being constructed on the heap!)
    vec_xorshift_plus(__m256i _s0, __m256i _s1)
    {
      if (!is_aligned(&s0, 32) || !is_aligned(&s1, 32))
          throw std::runtime_error("Fatal: unaligned vec_xorshift_plus!  See discussion in rf_kernels.hpp");

        s0 = _s0;
        s1 = _s1;
    }

    // Initialize or load the state from bonsai/online_mask_filler.cpp at the start of the kernel
    vec_xorshift_plus(const uint64_t rng_state[8])
    {

        if (!is_aligned(&s0, 32) || !is_aligned(&s1, 32))
	    throw std::runtime_error("Fatal: unaligned vec_xorshift_plus!  See discussion in rf_kernels.hpp");

        s0 = _mm256_loadu_si256((const __m256i *) &rng_state[0]);
        s1 = _mm256_loadu_si256((const __m256i *) &rng_state[4]);   // 256-bit offset
    }

    
    // Store the rng state back in to bonsai/online_mask_filler.cpp when the kernel finished executing
    inline void store_state(uint64_t rng_state[8])
    {
        _mm256_storeu_si256((__m256i *) &rng_state[0], s0);
        _mm256_storeu_si256((__m256i *) &rng_state[4], s1);
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


    // This exists solely for the unit test of the online mask filler!
    // This is just gen_floats with extra pfailv1 and pallzero parameters that 
    // dictate whether a group of 32 random numbers (which it generates at once) should result
    // in a failed v1 estimate (i.e. >=24 weights less than the cutoff) or whether
    // all weight values should be zero. Currently, the implementation is a little 
    // boneheaded and can definitely be imporved...
    inline void gen_weights(float *weights, float pfailv1, float pallzero)
    {
        // If the first random number generated is less than pallzero, we make it all zero.
        // If the first randon number generated is less than pallzero + pfailv1 but
        // greater than pallzero, we make sure the v1 fails by distributing at least 
        // 24 zeros throughout the vector. Else, we just fill weights randomly!
        
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


}  // namespace rf_kernels

#endif  // _RF_KERNELS_XORSHIFT_PLUSHPP
