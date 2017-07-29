#include <random>

#ifndef _RF_KERNELS_HPP
#define _RF_KERNELS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


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
			     xorshift_plus &rng);


}  // namespace rf_kernels

#endif  // _RF_KERNELS_HPP
