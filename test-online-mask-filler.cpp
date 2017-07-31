// g++ -std=c++11 -Wall -O3 -L. -L$HOME/lib -lrf_kernels -march=native -o test-online-mask-filler test-online-mask-filler.cpp
// if export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$HOME/lib

#include <iostream>
#include <cassert>
#include <random>
#include "rf_kernels.hpp"

using namespace std;
using namespace rf_kernels;


// Unit test code! 
void print_vec(float *a)
{
  for (int i=0; i < 32; i++)
    cout << a[i] << " ";
  cout << "\n\n";
}

inline bool equality_checker(float a, float b, float epsilon)
{
  // I know this isn't great, but I think it should be sufficient
  return abs(a-b) < epsilon;
}

inline bool test_filler(int nfreq, int nt_chunk, float pfailv1, float pallzero, float w_cutoff=0.5, float w_clamp=3e-3, float var_clamp_add=3e-3, 
			float var_clamp_mult=3e-3, float var_weight=2e-3, int niter=10000)
{
  assert (nfreq * nt_chunk % 8 == 0);
  assert (nt_chunk % 8 == 0);
  assert (pfailv1 < 1);
  assert (pfailv1 >=0);
  assert (pallzero < 1);
  assert (pallzero >=0);
  random_device rd;

  for (int iter=0; iter<niter; iter++)
    {
      // First, we randomize the weights and intensity values
      // We need two copies to put through each processing function and compare
      xorshift_plus rn;
    
      float intensity[nfreq * nt_chunk];
      float weights[nfreq * nt_chunk];
      float intensity2[nfreq * nt_chunk];
      float weights2[nfreq * nt_chunk];

      // Generate intensities 8 at a time using vanilla prng
      for (int i=0; i < nfreq * nt_chunk; i+=8)
        rn.gen_floats(intensity + i);

      // Use custom function to generate weights 
      for (int i=0; i < nfreq * nt_chunk; i+=32)
	rn.gen_weights(weights + i, pfailv1, pallzero);

      // Copy
      for (int i=0; i < nfreq * nt_chunk; i++)
	{
	  intensity2[i] = intensity[i];
	  weights2[i] = weights[i];
	}

      // Now, we generate random values for the running variance and running weights
      // Using the vec_xorshift_plus functions was too much of a hassle due to vector issues
      // so I've opted to use the C++ rng suite
      mt19937 gen (rd());
      // Weights between 0 and 1
      uniform_real_distribution<float> w_dis(0.0, 1.0);
      // Variance between 0.0 and 0.02
      uniform_real_distribution<float> var_dis(0.0, 0.02);
    
      vector<float> running_var(nfreq);
      vector<float> running_var2(nfreq);
      vector<float> running_weights(nfreq);
      vector<float> running_weights2(nfreq);

      // Make two copies
      for (int i=0; i<nfreq; i++)
	{
	  running_var[i] = var_dis(gen);
	  running_var2[i] = running_var[i];
	  running_weights[i] = w_dis(gen);
	  running_weights2[i] = running_weights[i];
	}
    
      // As in the prng unit test, we need to ensure both random number generators are initialized with the same seed values!
      unsigned int rn1 = rd();
      unsigned int rn2 = rd();
      unsigned int rn3 = rd();
      unsigned int rn4 = rd();
      unsigned int rn5 = rd();
      unsigned int rn6 = rd();
      unsigned int rn7 = rd();
      unsigned int rn8 = rd();
      vec_xorshift_plus vec_rn(_mm256_setr_epi64x(1, 3, 5, 7), _mm256_setr_epi64x(2, 4, 6, 8));
      xorshift_plus sca_rn(1, 2, 3, 4, 5, 6, 7, 8);
      //vec_xorshift_plus vec_rn(_mm256_setr_epi64x(rn1, rn3, rn5, rn7), _mm256_setr_epi64x(rn2, rn4, rn6, rn8));
      //xorshift_plus sca_rn(rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8);
      online_mask_filler_params params{};    

      // Process away! Note that the double instances of nt_chunk are for the "stride" parameter which is equal to nt_chunk for this test
      float *running_weights_arr = &running_weights[0];
      float *running_var_arr = &running_var[0];
      float *running_weights_arr2 = &running_weights2[0];
      float *running_var_arr2 = &running_var2[0];
      online_mask_fill(params, nfreq, nt_chunk, nt_chunk, intensity, weights, running_var_arr, running_weights_arr, vec_rn);
      scalar_online_mask_fill(params, nfreq, nt_chunk, nt_chunk, intensity2, weights2, running_var_arr2, running_weights_arr2, sca_rn);

      // I realize this next bit isn't the most effecient possible way of doing this comparison, but I think this order will be helpful
      // for debugging any future errors! So it's easy to see where things have gone wrong!
      for (int ifreq=0; ifreq<nfreq; ifreq++)
	{
	  // Check running variance
	  if (!equality_checker(running_var[ifreq], running_var2[ifreq], 10e-8))
	    {
	      cout << "Something's gone wrong! The running variances at frequency " << ifreq << " on iteration " << iter << " are unequal!" << endl;
	      cout << "Scalar output: " << running_var2[ifreq] << "\t\t Vectorized output: " << running_var[ifreq] << endl;
	      return false;
	    }

	  // Check running weights
	  if (!equality_checker(running_weights[ifreq], running_weights2[ifreq], 10e-8))
	    {
	      cout << "Something's gone wrong! The running weights at frequency " << ifreq << " on iteration " << iter << " are unequal!" << endl;
	      cout << "Scalar output: " << running_weights2[ifreq] << "\t\t Vectorized output: " << running_weights[ifreq] << endl;
	      return false;
	    }

	  for (int i=0; i<nt_chunk; i++)
	    {
	      // Check intensity
	      if (!equality_checker(intensity[ifreq * nt_chunk + i], intensity2[ifreq * nt_chunk + i], 10e-5))
		{ 
		  cout << "Something has gone wrong! The intensity array produced by the scalar mask filler does not match the intensity array produced by the vectorized mask filler!" << endl;
		  cout << "Output terminated at time index " << i << " and frequency " << ifreq << " on iteration " << iter << endl;
		  cout << "Scalar output: " << intensity2[ifreq * nt_chunk + i] << "\t\t Vectorized output: " << intensity[ifreq * nt_chunk + i] << endl;
		  return false;
		}
	          
	      // Check weights
	      if (!equality_checker(weights[ifreq * nt_chunk + i], weights2[ifreq * nt_chunk + i], 10e-5))
		{
		  cout << "Something has gone wrong! The weights array produced by the scalar mask filler does not match the weights array produced by the vectorized mask filler!" << endl;
		  cout << "Output terminated at time index " << i << " and frequency " << ifreq << " on iteration " << iter << endl;
		  cout << "Scalar output: " << weights2[ifreq * nt_chunk + i] << "\t\t Vectorized output: " << weights[ifreq * nt_chunk + i] << endl;
		  return false;
		}
	    }
	}
    }
  cout << "***online_mask_filler unit test passed!" << endl;
  return true;
}


inline bool test_xorshift(int niter=10000)
{
    std::random_device rd;
    for (int iter=0; iter < niter; iter++)
    {
	// Make sure both prngs are initialized with the same random seeds
	unsigned int rn1 = rd();
	unsigned int rn2 = rd();
	unsigned int rn3 = rd();
	unsigned int rn4 = rd();
	unsigned int rn5 = rd();
	unsigned int rn6 = rd();
	unsigned int rn7 = rd();
	unsigned int rn8 = rd();
	
	vec_xorshift_plus a(_mm256_setr_epi64x(rn1, rn3, rn5, rn7), _mm256_setr_epi64x(rn2, rn4, rn6, rn8));
	float vrn_vec[8];
	
	xorshift_plus b(rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8);
	float srn_vec[8];
	
	__m256 vrn = a.gen_floats();
	_mm256_storeu_ps(&vrn_vec[0], vrn);
	b.gen_floats(srn_vec);

	for (int i=0; i<8; i++)
	{
	    if (!equality_checker(srn_vec[i], vrn_vec[i], 10e-7))
	    {
		cout << "S code outputs: ";
		print_vec(srn_vec);
		cout << "V code outputs: ";
		print_vec(vrn_vec);
		cout << "rng test failed on iteration " << iter << " and index " << i << ": scalar and vectorized prngs are out of sync!" << endl;
		return false;
	    }
	}
    }
    
    cout << "***vec_xorshift_plus unit test passed!" << endl;
    return true;
}


void run_online_mask_filler_unit_tests()
{
  // Externally-visible function for unit testing
  test_xorshift();
  test_filler(8, 32, 0.20, 0.20);
}


int main(int argc, char **argv)
{
    run_online_mask_filler_unit_tests();
    return 0;
}

