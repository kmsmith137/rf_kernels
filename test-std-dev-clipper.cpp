#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/mean_rms.hpp"
#include "rf_kernels/std_dev_clipper.hpp"

using namespace std;
using namespace rf_kernels;


// reference_std_dev_clipper has been simplified so that it just does the following:
//   - rf_kernels::weighted_mean_rms()
//   - std_dev_clipper::_clip_1d()
//   - upsample mask using ad hoc code
//
// This assumes correctness of rf_kernels::weighted_mean_rms(), but this has its own unit test.
//
// Note that correctness of std_dev_clipper::_clip_1d() isn't tested anywhere!
// I ended up not unit-testing this, since it's so simple that a reference implementation
// would be equivalent.


static void reference_std_dev_clipper(const float *intensity, float *weights, int nfreq, int nt, int stride, axis_type axis, double sigma, int Df, int Dt, bool two_pass)
{
    int nfreq_ds = xdiv(nfreq, Df);
    int nt_ds = xdiv(nt, Dt);

    rf_kernels::weighted_mean_rms wrms(nfreq, nt, axis, Df, Dt, 1, 0, two_pass);
    wrms.compute_wrms(intensity, weights, stride);
    
    rf_kernels::std_dev_clipper sd(nfreq, nt, axis, sigma, Df, Dt, two_pass);
    rf_assert(sd.ntmp_v == wrms.nout);

    for (int i = 0; i < sd.ntmp_v; i++)
	sd.tmp_v[i] = square(wrms.out_rms[i]);

    sd._clip_1d();

    if (axis == AXIS_FREQ) {
	rf_assert(sd.ntmp_v == nt_ds);

	for (int it_ds = 0; it_ds < nt_ds; it_ds++) {
	    if (sd.tmp_v[it_ds] > 0)
		continue;

	    for (int ifreq = 0; ifreq < nfreq; ifreq++)
		for (int it = it_ds*Dt; it < (it_ds+1)*Dt; it++)
		    weights[ifreq*stride + it] = 0.0;
	}
    }
    else {
	rf_assert(sd.ntmp_v == nfreq_ds);

	for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++) {
	    if (sd.tmp_v[ifreq_ds] > 0)
		continue;

	    for (int ifreq = (ifreq_ds)*Df; ifreq < (ifreq_ds+1)*Df; ifreq++)
		for (int it = 0; it < nt; it++)
		    weights[ifreq*stride + it] = 0.0;
	}
    }
}


static void test_std_dev_clipper(std::mt19937 &rng, int nfreq, int nt, int stride, axis_type axis, double sigma, int Df, int Dt, bool two_pass)
{
#if 0
    cout << "test_std_dev_clipper: nfreq=" << nfreq << ", nt=" << nt << ", stride=" << stride << ", axis=" << axis 
	 << ", sigma=" << sigma << ", Df=" << Df << ", Dt=" << Dt << ", two_pass=" << two_pass << endl;
#endif

    vector<float> in_i = vector<float> (nfreq * stride, 0.0);
    vector<float> in_w = vector<float> (nfreq * stride, 0.0);

    // Low value chosen to expose corner cases.
    float pnonzero = 1.0 / float(Df*Dt);

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt; it++) {
	    in_i[ifreq*stride+it] = uniform_rand(rng);
	    in_w[ifreq*stride+it] = (uniform_rand(rng) < pnonzero) ? uniform_rand(rng) : 0.0;
	}
    }

    // Copy weights before running clipper.
    vector<float> in_w2 = in_w;

    // Fast kernel.
    std_dev_clipper sd(nfreq, nt, axis, sigma, Df, Dt, two_pass);
    sd.clip(&in_i[0], &in_w[0], stride);

    // Reference kernels.
    reference_std_dev_clipper(&in_i[0], &in_w2[0], nfreq, nt, stride, axis, sigma, Df, Dt, two_pass);

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt; it++) {
	    int i = ifreq * stride + it;
	    rf_assert(in_w[i] == in_w2[i]);
	}
    }
}


static void test_std_dev_clipper(std::mt19937 &rng)
{
    for (int iter = 0; iter < 1000; iter++) {
	int Df = 1 << randint(rng, 0, 6);
	int Dt = 1 << randint(rng, 0, 6);
	int nfreq = Df * randint(rng, 1, 65);
	int nt = 8 * Dt * randint(rng, 1, 17);
	int stride = randint(rng, nt, 2*nt);
	axis_type axis = randint(rng, 0, 2) ? AXIS_FREQ : AXIS_TIME;
	double sigma = uniform_rand(rng, 1.1, 1.5);
	bool two_pass = randint(rng, 0, 2);
	
	// Round up to multiple of 8.
	nfreq = ((nfreq+7)/8) * 8;

	test_std_dev_clipper(rng, nfreq, nt, stride, axis, sigma, Df, Dt, two_pass);
    }

    cout << "test_std_dev_clipper: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_std_dev_clipper(rng);

    return 0;
}
