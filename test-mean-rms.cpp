#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/downsample.hpp"
#include "rf_kernels/mean_rms.hpp"

using namespace std;
using namespace rf_kernels;


inline void reference_weighted_mean(float &out_mean, float &out_rms, const float *i_in, const float *w_in, int n, int stride)
{
    float wsum = 0.0;
    float wisum = 0.0;
    float wiisum = 0.0;

    for (int i = 0; i < n; i++) {
	wsum += w_in[i*stride];
	wisum += w_in[i*stride] * i_in[i*stride];
    }

    out_mean = (wsum > 0.0) ? (wisum/wsum) : 0.0;

    for (int i = 0; i < n; i++)
	wiisum += w_in[i*stride] * square(i_in[i*stride] - out_mean);

    out_rms = (wsum > 0.0) ? sqrt(wiisum/wsum) : 0.0;
}


static void test_weighted_mean_rms(std::mt19937 &rng, int nfreq, int nt_chunk, int stride, axis_type axis, int Df, int Dt, int niter, double sigma, bool two_pass)
{
    rf_assert(axis == AXIS_TIME);
    
    int nfreq_ds = xdiv(nfreq, Df);
    int nt_ds = xdiv(nt_chunk, Dt);
    int nout = nfreq_ds;  // AXIS_TIME assumed
	
    rf_kernels::weighted_mean_rms wrms(nfreq, nt_chunk, axis, Df, Dt, niter, sigma, two_pass);
    
    rf_assert(wrms.nfreq_ds == nfreq_ds);
    rf_assert(wrms.nt_ds == nt_ds);
    rf_assert(wrms.nout == nout);

    // FIXME improve random generation of test data so that corner cases are exposed.
    vector<float> i_in = uniform_randvec(rng, nfreq*stride, -1.0, 1.0);
    vector<float> w_in = uniform_randvec(rng, nfreq*stride, 0.1, 1.0);

    wrms.compute_wrms(&i_in[0], &w_in[0], stride);

    // Outputs from reference wrms
    vector<float> i_ds(nfreq_ds * nt_ds, 0.0);
    vector<float> w_ds(nfreq_ds * nt_ds, 0.0);
    vector<float> mean(nout, 0.0);
    vector<float> rms(nout, 0.0);
	
    // Reference weighted_mean_rms
    // Step 1: downsample
    rf_kernels::wi_downsampler ds(Df, Dt);
    ds.downsample(nfreq_ds, nt_ds, &i_ds[0], &w_ds[0], nt_ds, &i_in[0], &w_in[0], stride);

    // Step 2: compute wrms (AXIS_TIME assumed)
    for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++)
	reference_weighted_mean(mean[ifreq_ds], rms[ifreq_ds], &i_ds[ifreq_ds*nt_ds], &w_ds[ifreq_ds*nt_ds], nt_ds, 1);

    // Now compare fast vs reference
    
    for (int i = 0; i < nout; i++) {
	rf_assert(abs(wrms.out_mean[i] - mean[i]) < 1.0e-4);
	rf_assert(abs(wrms.out_rms[i] - rms[i]) < 1.0e-4);
    }

    for (int i = 0; i < wrms.nfreq_ds * wrms.nt_ds; i++)
	rf_assert(abs(wrms.tmp_i[i] - i_ds[i]) < 1.0e-4);

    for (int i = 0; i < wrms.nfreq_ds * wrms.nt_ds; i++)
	rf_assert(abs(wrms.tmp_w[i] - w_ds[i]) < 1.0e-4);
}


static void test_weighted_mean_rms(std::mt19937 &rng)
{
    const int niter = 1000;

    cout << "test_weighted_mean_rms: don't forget, unit test does not have full generality yet" << endl;
    
    for (int iter = 0; iter < niter; iter++) {
	int Df = 1 << randint(rng, 0, 2);  // FIXME
	int Dt = 1 << randint(rng, 0, 2);  // FIXME
	int nfreq = Df * randint(rng, 1, 17);
	int nt = 8 * Dt * randint(rng, 1, 17);	
	int stride = randint(rng, nt, 2*nt);
	axis_type axis = AXIS_TIME;  // FIXME
	int niter = 1;  // FIXME
	double sigma = 0.0;  // FIXME
	bool two_pass = true;  // FIXME

	test_weighted_mean_rms(rng, nfreq, nt, stride, axis, Df, Dt, niter, sigma, two_pass);
    }

    cout << "test_weighted_mean_rms: pass" << endl;
}	


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_weighted_mean_rms(rng);

    return 0;
}
