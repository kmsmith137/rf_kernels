#include <iostream>
#include <cassert>
#include <cstring>
#include <random>

#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/xorshift_plus.hpp"
#include "rf_kernels/online_mask_filler.hpp"

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
    return fabsf(a-b) < epsilon;
}



// gen_weights(): generates 32 random weights (one v1_chunk)
//
// This exists solely for the unit test of the online mask filler!
// The pallzero parameter dictates whether a group of 32 random numbers 
// (which it generates at once) should all be zero.
//
// KMS: this was previously a member function of 'struct xorshift_plus', but I switched
// to using std::mt19937, since I wanted to make some minor changes which were easier
// using the standard library RNG.


inline void gen_weights(std::mt19937 &rng, float *weights, float pallzero, float w_cutoff)
{
    // If the first random number generated is less than pallzero, we make it all zero.

    if (uniform_rand(rng) < pallzero) {
	for (int i = 0; i < 32; i++)
	    weights[i] = 0.0;
    }
    else {
	// Avoid generating weights too close to w_cutoff, unless w_cutoff=0
	for (int i = 0; i < 32; i++) {
	    if (uniform_rand(rng) < 0.5)
		weights[i] = uniform_rand(rng, 1.01*w_cutoff, 1.0);
	    else if (w_cutoff > 0.0)
		weights[i] = uniform_rand(rng, 0.0, 0.99*w_cutoff);
	    else
		weights[i] = 0.0;
	}
    }
}


void test_filler(std::mt19937 &rng, online_mask_filler &params, int nt_chunk, int ostride, int istride, int wstride, int niter)
{
    // Used when randomly generating weights below.
    const float pallzero = 0.2;

    // Assumed by gen_weights() below.
    assert (nt_chunk % 32 == 0);

    const int nfreq = params.nfreq;
    
    vector<float> intensity(nfreq * istride);   // mask_fill_in_place()
    vector<float> intensity2(nfreq * istride);  // mask_fill_and_multiply()
    vector<float> intensity3(nfreq * istride);  // scalar_mask_fill_in_place()

    vector<float> weights(nfreq * wstride);
    vector<float> weights2(nfreq * wstride);
    vector<float> weights3(nfreq * wstride);

    vector<float> out2(nfreq * ostride);
    
    // Weights between 0 and 1
    uniform_real_distribution<float> w_dis(0.0, 1.0);
    // Variance between 0.0 and 0.02
    uniform_real_distribution<float> var_dis(0.0, 0.02);
    
    for (int iter=0; iter<niter; iter++)
    {
        // First, we randomize the weights and intensity values
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    for (int it = 0; it < nt_chunk; it++)
		intensity[ifreq*istride + it] = uniform_rand(rng);
	
	// Use custom function to generate weights 
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    for (int it = 0; it < nt_chunk; it += 32)
		gen_weights(rng, &weights[ifreq*wstride + it], pallzero, params.w_cutoff);

        // We need two copies to put through each processing function and compare
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    for (int it = 0; it < nt_chunk; it++) {
		intensity2[ifreq*istride + it] = intensity[ifreq*istride + it];
		intensity3[ifreq*istride + it] = intensity[ifreq*istride + it];
		weights2[ifreq*wstride + it] = weights[ifreq*wstride + it];
		weights3[ifreq*wstride + it] = weights[ifreq*wstride + it];
	    }
	}

	// Note: copy constructor makes a "deep copy" (running_* fields and RNG state are copied)
	online_mask_filler params2 = params;
	online_mask_filler params3 = params;

	// Process away!  (Parts 1+2 out of 3)
	params.mask_fill_in_place(nt_chunk, &intensity[0], istride, &weights[0], wstride);
	params2.mask_fill_and_multiply(nt_chunk, &out2[0], ostride, &intensity2[0], istride, &weights2[0], wstride);

	// Check that mask_fill_and_multiply() didn't modify its input arrays.
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    for (int it = 0; it < nt_chunk; it++) {
		if (intensity2[ifreq*istride+it] != intensity3[ifreq*istride+it])
		    throw runtime_error("online_mask_filler::mask_fill_and_multiply() modified its input intensity array");
		if (weights2[ifreq*wstride+it] != weights3[ifreq*wstride+it])
		    throw runtime_error("online_mask_filler::mask_fill_and_multiply() modified its input weights array");
	    }
	}

	// Process away!  (Part 3 out of 3)
	params3.scalar_mask_fill_in_place(nt_chunk, &intensity3[0], istride, &weights3[0], wstride);

	float *running_var = params.running_var.get();
	float *running_var2 = params2.running_var.get();
	float *running_var3 = params3.running_var.get();
	float *running_weights = params.running_weights.get();
	float *running_weights2 = params2.running_weights.get();
	float *running_weights3 = params3.running_weights.get();

	for (int ifreq=0; ifreq<nfreq; ifreq++)
	{
	    // Check running variance
	    if (!equality_checker(running_var[ifreq], running_var3[ifreq], 1.0e-6) || !equality_checker(running_var2[ifreq], running_var3[ifreq], 1.0e-6))
	    {
		cout << "Something's gone wrong! The running variances at frequency " << ifreq << " on iteration " << iter << " are unequal!" << endl;
		cout << "mask_fill_in_place=" << running_var[ifreq] << ", mask_fill_and_multiply=" << running_var2[ifreq] << ", scalar_mask_fill_in_place=" << running_var3[ifreq] << endl;
		exit(1);
	    }

	    // Check running weights
	    if (!equality_checker(running_weights[ifreq], running_weights3[ifreq], 1.0e-6) || !equality_checker(running_weights2[ifreq], running_weights3[ifreq], 1.0e-6))
	    {
		cout << "Something's gone wrong! The running weights at frequency " << ifreq << " on iteration " << iter << " are unequal!" << endl;
		cout << "mask_fill_in_place=" << running_weights[ifreq] << ", mask_fill_and_multiply=" << running_weights2[ifreq] << ", scalar_mask_fill_in_place=" << running_weights3[ifreq] << endl;
		exit(1);
	    }
	    
	    for (int i=0; i<nt_chunk; i++)
	    {
		// Check intensity
		if (!equality_checker(intensity[ifreq * istride + i], intensity3[ifreq * istride + i], 1.0e-6))
		{ 
		    cout << "Something has gone wrong! The intensity array produced by the scalar mask filler does not match the intensity array produced by the vectorized mask filler!" << endl;
		    cout << "Output terminated at time index " << i << " and frequency " << ifreq << " on iteration " << iter << endl;
		    cout << "Scalar output: " << intensity3[ifreq * istride + i] << "\t\t Vectorized output: " << intensity[ifreq * istride + i] << endl;
		    exit(1);
		}
		
		// Check weights
		if (!equality_checker(weights[ifreq * wstride + i], weights3[ifreq * wstride + i], 1.0e-6))
		{
		    cout << "Something has gone wrong! The weights array produced by the scalar mask filler does not match the weights array produced by the vectorized mask filler!" << endl;
		    cout << "Output terminated at time index " << i << " and frequency " << ifreq << " on iteration " << iter << endl;
		    cout << "Scalar output: " << weights3[ifreq * wstride + i] << "\t\t Vectorized output: " << weights[ifreq * wstride + i] << endl;
		    exit(1);
		}

		// Check output of mask_fill_and_multiply()
		if (!equality_checker(out2[ifreq*ostride+i], intensity3[ifreq*istride+i] * weights3[ifreq*wstride+i], 1.0e-6)) {
		    cout << "Something has gone wrong! The output of mask_fill_and_multiply() is not consistent with scalar_mask_fill_in_place()" << endl;
		    cout << "Output terminated at time index " << i << " and frequency " << ifreq << " on iteration " << iter << endl;
		    cout << "mask_fill_and_multiply=" << out2[ifreq*ostride+i] << ", scalar equivalent = " << (intensity3[ifreq*istride+i] * weights3[ifreq*wstride+i]) << endl;
		    exit(1);
		}
	    }
	}
    }
}


void test_filler(int nouter=100)
{
    // Using the vec_xorshift_plus functions was too much of a hassle due to vector issues
    // so I've opted to use the C++ rng suite to generate input data.
    random_device rd;
    mt19937 rng (rd());
 
    // In each "outer" iteration, the parameters of the unit test are randomized.
    for (int iouter = 0; iouter < nouter; iouter++) {
	int nfreq = randint(rng, 1, 65);
	int nt_chunk = randint(rng, 1, 9) * 32;   // must be multiple of v1_chunk=32
	int ostride = randint(rng, nt_chunk, 2*nt_chunk);
	int istride = randint(rng, nt_chunk, 2*nt_chunk);
	int wstride = randint(rng, nt_chunk, 2*nt_chunk);
	int ninner = 100;   // Number of "inner" iterations

	online_mask_filler params(nfreq);
	params.v1_chunk = 32;   // currently hardcoded
	params.var_weight = uniform_rand(rng, 1.0e-3, 0.1);
	params.w_clamp = uniform_rand(rng, 1.0e-10, 1.0);
	params.w_cutoff = randint(rng,0,4) ? uniform_rand(rng, 0.1, 0.9) : 0.0;

#if 0
	cout << "outer iteration " << iouter << endl;
#endif
	
	test_filler(rng, params, nt_chunk, ostride, istride, wstride, ninner);
    }

    cout << "***online_mask_filler unit test passed!" << endl;
}


inline bool test_xorshift(int niter=10000)
{
    std::random_device rd;
    for (int iter=0; iter < niter; iter++)
    {
	// Make sure both prngs are initialized with the same random seeds
	uint64_t rn1 = rd64(rd);
	uint64_t rn2 = rd64(rd);
	uint64_t rn3 = rd64(rd);
	uint64_t rn4 = rd64(rd);
	uint64_t rn5 = rd64(rd);
	uint64_t rn6 = rd64(rd);
	uint64_t rn7 = rd64(rd);
	uint64_t rn8 = rd64(rd);
	
	uint64_t rng_state[8]{rn1, rn3, rn5, rn7, rn2, rn4, rn6, rn8};
	vec_xorshift_plus a(rng_state);
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
		exit(1);
	    }
	}
    }
    
    cout << "***vec_xorshift_plus unit test passed!" << endl;
    return true;
}


int main(int argc, char **argv)
{
    test_xorshift();
    test_filler();

    return 0;
}
