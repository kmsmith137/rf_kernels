// #include <simd_helpers/simd_debug.hpp>

#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/downsample.hpp"

using namespace std;
using namespace rf_kernels;


// -------------------------------------------------------------------------------------------------
//
// test_wi_downsample


static void reference_wi_downsample(int nfreq_out, int nt_out, float *out_i, int out_istride, float *out_w, int out_wstride,
				    const float *in_i, int in_istride, const float *in_w, int in_wstride, int Df, int Dt)
{
    // No argument checking!

    for (int ifreq_out = 0; ifreq_out < nfreq_out; ifreq_out++) {
	for (int it_out = 0; it_out < nt_out; it_out++) {
	    float wisum = 0.0;
	    float wsum = 0.0;

	    for (int ifreq_in = ifreq_out*Df; ifreq_in < (ifreq_out+1)*Df; ifreq_in++) {
		for (int it_in = it_out*Dt; it_in < (it_out+1)*Dt; it_in++) {
		    float ival = in_i[ifreq_in * in_istride + it_in];
		    float wval = in_w[ifreq_in * in_wstride + it_in];

		    wisum += wval * ival;
		    wsum += wval;
		}
	    }

	    out_i[ifreq_out * out_istride + it_out] = (wsum > 0.0) ? (wisum/wsum) : 0.0;
	    out_w[ifreq_out * out_wstride + it_out] = wsum;
	}
    }
}


static void test_wi_downsample(std::mt19937 &rng, int nfreq_out, int nt_out, int out_istride, int out_wstride, int in_istride, int in_wstride, int Df, int Dt)
{
    int nfreq_in = nfreq_out * Df;
    vector<float> in_i = rf_kernels::uniform_randvec(rng, nfreq_in * in_istride, 0.0, 1.0);
    vector<float> in_w = vector<float> (nfreq_in * in_wstride, 0.0);

    // Low value chosen to expose corner cases.
    float pnonzero = 1.0 / float(Df*Dt); 
    for (size_t i = 0; i < in_w.size(); i++)
	in_w[i] = (uniform_rand(rng) < pnonzero) ? uniform_rand(rng) : 0.0;

    vector<float> out_i1 = rf_kernels::uniform_randvec(rng, nfreq_out * out_istride, 0.0, 1.0);
    vector<float> out_i2 = rf_kernels::uniform_randvec(rng, nfreq_out * out_istride, 0.0, 1.0);
    vector<float> out_w1 = rf_kernels::uniform_randvec(rng, nfreq_out * out_wstride, 0.0, 1.0);
    vector<float> out_w2 = rf_kernels::uniform_randvec(rng, nfreq_out * out_wstride, 0.0, 1.0);

    wi_downsampler ds(Df, Dt);
    ds.downsample(nfreq_out, nt_out, &out_i1[0], out_istride, &out_w1[0], out_wstride, &in_i[0], in_istride, &in_w[0], in_wstride);
    reference_wi_downsample(nfreq_out, nt_out, &out_i2[0], out_istride, &out_w2[0], out_wstride, &in_i[0], in_istride, &in_w[0], in_wstride, Df, Dt);					     

#if 0
    // Uncomment for an avalanche of debugging output!
    for (int ifreq_out = 0; ifreq_out < nfreq_out; ifreq_out++) {
	for (int it_out = 0; it_out < nt_out; it_out++) {
	    cout << "ifreq_out=" << ifreq_out << ", it_out=" << it_out
		 << ": fast=(" << out_i1[ifreq_out*out_istride+it_out] << "," << out_w1[ifreq_out*out_wstride+it_out] 
		 << "), slow=(" << out_i2[ifreq_out*out_istride+it_out] << "," << out_w2[ifreq_out*out_wstride+it_out] 
		 << ")\n";

	    for (int ifreq_in = ifreq_out*Df; ifreq_in < (ifreq_out+1)*Df; ifreq_in++) {
		cout << "    ";
		for (int it_in = it_out*Dt; it_in < (it_out+1)*Dt; it_in++)
		    cout << " (" << in_i[ifreq_in*in_istride+it_in] << "," << in_w[ifreq_in*in_istride+it_in] << ")";
		cout << endl;
	    }
	}
    }
#endif
    
    for (int ifreq_out = 0; ifreq_out < nfreq_out; ifreq_out++) {
	for (int it_out = 0; it_out < nt_out; it_out++) {
	    int ii = ifreq_out * out_istride + it_out;
	    int wi = ifreq_out * out_wstride + it_out;
	    rf_assert(fabs(out_i1[ii]-out_i2[ii]) < 1.0e-4);
	    rf_assert(fabs(out_w1[wi]-out_w2[wi]) < 1.0e-4);
	}
    }
}


static void test_wi_downsample(std::mt19937 &rng)
{
    const int niter = 300;
    cout << "test_wi_downsample: start" << endl;
    
    for (int iter = 0; iter < niter; iter++) {
	int Df = 1 << randint(rng, 0, 7);
	int Dt = 1 << randint(rng, 0, 7);
	
	int nfreq_out = randint(rng, 1, 17);
	int nt_out = 8 * randint(rng, 1, 17);
	int out_istride = randint(rng, nt_out, 2*nt_out);
	int out_wstride = randint(rng, nt_out, 2*nt_out);
	int in_istride = randint(rng, Dt*nt_out, 2*Dt*nt_out);
	int in_wstride = randint(rng, Dt*nt_out, 2*Dt*nt_out);
	
	test_wi_downsample(rng, nfreq_out, nt_out, out_istride, out_wstride, in_istride, in_wstride, Df, Dt);
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
