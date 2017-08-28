// #include <simd_helpers/simd_debug.hpp>

#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/downsample.hpp"
#include "rf_kernels/downsample_internals.hpp"

using namespace std;
using namespace rf_kernels;
using namespace simd_helpers;



// -------------------------------------------------------------------------------------------------
//
// test_wi_downsample


static void reference_wi_downsample(int nfreq_out, int nt_out, float *out_i, float *out_w, int ostride,
				    const float *in_i, const float *in_w, int istride, int Df, int Dt)
{
    // No argument checking!

    for (int ifreq_out = 0; ifreq_out < nfreq_out; ifreq_out++) {
	for (int it_out = 0; it_out < nt_out; it_out++) {
	    float wisum = 0.0;
	    float wsum = 0.0;

	    for (int ifreq_in = ifreq_out*Df; ifreq_in < (ifreq_out+1)*Df; ifreq_in++) {
		for (int it_in = it_out*Dt; it_in < (it_out+1)*Dt; it_in++) {
		    float ival = in_i[ifreq_in*istride + it_in];
		    float wval = in_w[ifreq_in*istride + it_in];

		    wisum += wval * ival;
		    wsum += wval;
		}
	    }

	    out_i[ifreq_out*ostride + it_out] = (wsum > 0.0) ? (wisum/wsum) : 0.0;
	    out_w[ifreq_out*ostride + it_out] = wsum;
	}
    }
}


static void test_wi_downsample(std::mt19937 &rng, int nfreq_out, int nt_out, int ostride, int istride, int Df, int Dt)
{
    int nfreq_in = nfreq_out * Df;
    vector<float> in_i = uniform_randvec(rng, nfreq_in * istride, 0.0, 1.0);
    vector<float> in_w = vector<float> (nfreq_in * istride, 0.0);

    // Low value chosen to expose corner cases.
    float pnonzero = 1.0 / float(Df*Dt); 
    for (size_t i = 0; i < in_w.size(); i++)
	in_w[i] = (uniform_rand(rng) < pnonzero) ? uniform_rand(rng) : 0.0;

    vector<float> out_i1 = uniform_randvec(rng, nfreq_out * ostride, 0.0, 1.0);
    vector<float> out_i2 = uniform_randvec(rng, nfreq_out * ostride, 0.0, 1.0);
    vector<float> out_w1 = uniform_randvec(rng, nfreq_out * ostride, 0.0, 1.0);
    vector<float> out_w2 = uniform_randvec(rng, nfreq_out * ostride, 0.0, 1.0);

    wi_downsampler ds(Df, Dt);
    ds.downsample(nfreq_out, nt_out, &out_i1[0], &out_w1[0], ostride, &in_i[0], &in_w[0], istride);
    reference_wi_downsample(nfreq_out, nt_out, &out_i2[0], &out_w2[0], ostride, &in_i[0], &in_w[0], istride, Df, Dt);					     
    
    for (int ifreq_out = 0; ifreq_out < nfreq_out; ifreq_out++) {
	for (int it_out = 0; it_out < nt_out; it_out++) {
	    int i = ifreq_out * ostride + it_out;
	    rf_assert(fabs(out_i1[i]-out_i2[i]) < 1.0e-4);
	    rf_assert(fabs(out_w1[i]-out_w2[i]) < 1.0e-4);
	}
    }
}


static void test_wi_downsample(std::mt19937 &rng)
{
    for (int iter = 0; iter < 1000; iter++) {
	int Df = 1 << randint(rng, 0, 5);
	int Dt = 1 << randint(rng, 0, 5);
	
	int nfreq_out = randint(rng, 1, 17);
	int nt_out = 8 * randint(rng, 1, 17);
	int ostride = randint(rng, nt_out, 2*nt_out);
	int istride = randint(rng, Dt*nt_out, 2*Dt*nt_out);
	
	test_wi_downsample(rng, nfreq_out, nt_out, ostride, istride, Df, Dt);
    }

    cout << "test_wi_downsample: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_wi_downsample(rng);

    return 0;
}
