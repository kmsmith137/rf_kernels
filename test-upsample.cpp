#include <simd_helpers/simd_debug.hpp>

#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/upsample.hpp"

using namespace std;
using namespace rf_kernels;
using namespace simd_helpers;


// -------------------------------------------------------------------------------------------------
//
// test_weight_upsample


static void reference_weight_upsample(int nfreq_in, int nt_in, float *out, int ostride, const float *in, int istride, float w_cutoff, int Df, int Dt)
{
    // No argument checking!

    for (int ifreq = 0; ifreq < nfreq_in; ifreq++) {
	for (int it = 0; it < nt_in; it++) {
	    float w = in[ifreq*istride + it];
	    if (w > w_cutoff)
		continue;

	    for (int ifreq_out = ifreq*Df; ifreq_out < (ifreq+1)*Df; ifreq_out++)
		for (int it_out = it*Dt; it_out < (it+1)*Dt; it_out++)
		    out[ifreq_out*ostride + it_out] = 0.0;
	}
    }
}


static void test_weight_upsample(std::mt19937 &rng, int nfreq_in, int nt_in, int ostride, int istride, float w_cutoff, int Df, int Dt)
{
    vector<float> w_in(nfreq_in * istride);
    for (size_t i = 0; i < w_in. size(); i++)
	w_in[i] = (uniform_rand(rng) < 0.5) ? uniform_rand(rng) : 0.0;
    
    vector<float> w_out1(Df * nfreq_in * ostride);
    for (size_t i = 0; i < w_out1.size(); i++)
	w_out1[i] = (uniform_rand(rng) < 0.5) ? uniform_rand(rng) : 0.0;

    vector<float> w_out2 = w_out1;

    weight_upsampler u(Df, Dt);
    u.upsample(nfreq_in, nt_in, &w_out1[0], ostride, &w_in[0], istride, w_cutoff);    
    reference_weight_upsample(nfreq_in, nt_in, &w_out2[0], ostride, &w_in[0], istride, w_cutoff, Df, Dt);

    for (size_t i = 0; i < w_out1.size(); i++)
	rf_assert(w_out1[i] == w_out2[i]);
}


static void test_weight_upsample(std::mt19937 &rng)
{
    for (int iter = 0; iter < 1000; iter++) {
	int Df = 1 << randint(rng, 0, 7);
	int Dt = 1 << randint(rng, 0, 7);
	
	int nfreq_in = randint(rng, 1, 17);
	int nt_in = 8 * randint(rng, 1, 17);
	int ostride = randint(rng, Dt*nt_in, 2*Dt*nt_in);
	int istride = randint(rng, nt_in, 2*nt_in);
	float w_cutoff = max(0.0, uniform_rand(rng,-1.0,1.0));
	
	test_weight_upsample(rng, nfreq_in, nt_in, ostride, istride, w_cutoff, Df, Dt);
    }

    cout << "test_weight_upsample: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_weight_upsample(rng);

    return 0;
}
