#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"

#include "rf_kernels/mean_rms.hpp"
#include "rf_kernels/upsample.hpp"
#include "rf_kernels/downsample.hpp"
#include "rf_kernels/intensity_clipper.hpp"

using namespace std;
using namespace rf_kernels;


// -------------------------------------------------------------------------------------------------


// Reference weighted_mean_and_rms take 1: assumes (axis,Df,Dt,niter)=(None,1,1,1).
static void _ref_wrms(float *out_mean, float *out_rms, int nfreq, int nt, const float *i_in, const float *w_in, int stride)
{
    float wsum = 0.0;
    float wisum = 0.0;
    float wiisum = 0.0;

    // First pass
    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt; it++) {
	    wsum += w_in[ifreq*stride + it];
	    wisum += w_in[ifreq*stride + it] * i_in[ifreq*stride + it];
	}
    }

    float mean = (wsum > 0) ? (wisum / wsum) : 0.0;

    // Second pass (note that there is no need to recompute wsum)
    for (int ifreq = 0; ifreq < nfreq; ifreq++)
	for (int it = 0; it < nt; it++)
	    wiisum += w_in[ifreq*stride + it] * square(i_in[ifreq*stride + it] - mean);

    *out_mean = mean;
    *out_rms = (wsum > 0) ? sqrt(wiisum/wsum) : 0.0;
}


// Reference weighted_mean_and_rms take 2: (Df,Dt,niter)=(1,1,1) assumed, but arbitrary axis allowed.
static void reference_wrms(float *out_mean, float *out_rms, int nfreq, int nt, axis_type axis, const float *i_in, const float *w_in, int stride)
{
    if (axis == AXIS_FREQ) {
	for (int it = 0; it < nt; it++)
	    _ref_wrms(out_mean + it, out_rms + it, nfreq, 1, i_in + it, w_in + it, stride);
    }
    else if (axis == AXIS_TIME) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    _ref_wrms(out_mean + ifreq, out_rms + ifreq, 1, nt, i_in + ifreq*stride, w_in + ifreq*stride, 0);
    }
    else if (axis == AXIS_NONE)
	_ref_wrms(out_mean, out_rms, nfreq, nt, i_in, w_in, stride);
    else
	throw runtime_error("bad axis in reference_wrms()");
}


// -------------------------------------------------------------------------------------------------


// Helper for reference_iclip: assumes (axis,Df,Dt)=(None,1,1) and (mean,thresh) already computed.
static void _ref_iclip(int nfreq, int nt, const float *i_in, float *w_in, int stride, float mean, float thresh)
{
    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt; it++) {
	    if (abs(i_in[ifreq*stride+it] - mean) > thresh)
		w_in[ifreq*stride+it] = 0.0;
	}
    }
}


// reference_iclip: assumes (Df,Dt)=(1,1) and (mean,rms) precomputed, but allows arbitrary axis.
static void reference_iclip(int nfreq, int nt, axis_type axis, const float *i_in, float *w_in, int stride, const float *mean, const float *rms, float sigma)
{
    if (axis == AXIS_FREQ) {
	for (int it = 0; it < nt; it++)
	    _ref_iclip(nfreq, 1, i_in + it, w_in + it, stride, mean[it], sigma * rms[it]);
    }
    else if (axis == AXIS_TIME) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    _ref_iclip(1, nt, i_in + ifreq*stride, w_in + ifreq*stride, 0, mean[ifreq], sigma * rms[ifreq]);
    }
    else if (axis == AXIS_NONE)
	_ref_iclip(nfreq, nt, i_in, w_in, stride, mean[0], sigma * rms[0]);
    else
	throw runtime_error("bad axis in reference_iclip()");
}


// -------------------------------------------------------------------------------------------------


static void test_wrms(std::mt19937 &rng, int nfreq, int nt_chunk, int stride, axis_type axis, int Df, int Dt, int niter, double sigma, bool two_pass)
{
    int nfreq_ds = xdiv(nfreq, Df);
    int nt_ds = xdiv(nt_chunk, Dt);
    
    int nout = 1;
    if (axis == AXIS_FREQ) nout = nt_ds;
    if (axis == AXIS_TIME) nout = nfreq_ds;

    // FIXME improve random generation of test data so that corner cases are exposed.
    vector<float> i_in = uniform_randvec(rng, nfreq*stride, -1.0, 1.0);
    vector<float> w_in = uniform_randvec(rng, nfreq*stride, 0.1, 1.0);

    rf_kernels::weighted_mean_rms wrms(nfreq, nt_chunk, axis, Df, Dt, niter, sigma, two_pass);    
    rf_assert(wrms.nfreq_ds == nfreq_ds);
    rf_assert(wrms.nt_ds == nt_ds);
    rf_assert(wrms.nout == nout);

    // Run fast wrms kernel.
    // Reminder: the outputs are stored in wrms.out_mean[] and wrms.out_rms[].
    wrms.compute_wrms(&i_in[0], &w_in[0], stride);

    // The rest of this routine is devoted to computing something to compare to!
    // Outputs from reference wrms will go here.
    vector<float> ref_mean(nout, 0.0);
    vector<float> ref_rms(nout, 0.0);

    // Temporary buffers.
    vector<float> i_ds(nfreq_ds * nt_ds, 0.0);
    vector<float> w_ds(nfreq_ds * nt_ds, 0.0);
	
    // Step 1: downsample
    rf_kernels::wi_downsampler ds(Df, Dt);
    ds.downsample(nfreq_ds, nt_ds, &i_ds[0], &w_ds[0], nt_ds, &i_in[0], &w_in[0], stride);

    // Step 2: account for the first (niter-1) iterations by clipping.
    if (niter > 1) {
	rf_kernels::intensity_clipper ic(nfreq_ds, nt_ds, axis, sigma, 1, 1, niter-1, 0, two_pass);
	ic.clip(&i_ds[0], &w_ds[0], nt_ds);
    }

    // Step 3: run reference wrms kernel.  Note that the reference kernel gets to assume (Df,Dt,niter) = (1,1,1).
    reference_wrms(&ref_mean[0], &ref_rms[0], nfreq_ds, nt_ds, axis, &i_ds[0], &w_ds[0], nt_ds);

    // Step 4: compare outputs!    
    for (int i = 0; i < nout; i++) {
	rf_assert(abs(wrms.out_mean[i] - ref_mean[i]) < 1.0e-4);
	rf_assert(abs(wrms.out_rms[i] - ref_rms[i]) < 1.0e-4);
    }
}


// -------------------------------------------------------------------------------------------------


static void test_intensity_clipper(std::mt19937 &rng, int nfreq, int nt_chunk, int stride, axis_type axis, int Df, int Dt, int niter, double sigma, double iter_sigma, bool two_pass)
{
    int nfreq_ds = xdiv(nfreq, Df);
    int nt_ds = xdiv(nt_chunk, Dt);
    
    // FIXME improve random generation of test data so that corner cases are exposed.
    vector<float> i_in = uniform_randvec(rng, nfreq*stride, -1.0, 1.0);
    vector<float> w_in = uniform_randvec(rng, nfreq*stride, 0.1, 1.0);

    rf_kernels::intensity_clipper ic(nfreq, nt_chunk, axis, sigma, Df, Dt, niter, iter_sigma, two_pass);
    rf_assert(ic.nfreq_ds == nfreq_ds);
    rf_assert(ic.nt_ds == nt_ds);

    // Run fast kernel.
    vector<float> w_fast = w_in;
    ic.clip(&i_in[0], &w_fast[0], stride);

    // The rest of this routine is devoted to computing something to compare to!

    // Temporary buffers.
    vector<float> i_ds(nfreq_ds * nt_ds, 0.0);
    vector<float> w_ds(nfreq_ds * nt_ds, 0.0);

    // Step 1: downsample.
    rf_kernels::wi_downsampler ds(Df, Dt);
    ds.downsample(nfreq_ds, nt_ds, &i_ds[0], &w_ds[0], nt_ds, &i_in[0], &w_in[0], stride);

    // Step 2: compute wrms.
    // Reminder: the outputs are stored in wrms.out_mean[] and wrms.out_rms[].
    rf_kernels::weighted_mean_rms wrms(nfreq_ds, nt_ds, axis, 1, 1, niter, iter_sigma, two_pass);
    wrms.compute_wrms(&i_ds[0], &w_ds[0], stride);

    // Step 3: run reference intensity clipper.
    // We do this twice, with slightly different thresholds, for numerical stability.
    vector<float> w_ds1 = w_ds;
    vector<float> w_ds2 = w_ds;

    reference_iclip(nfreq_ds, nt_ds, axis, &i_ds[0], &w_ds1[0], nt_ds, wrms.out_mean, wrms.out_rms, 0.9999 * sigma);
    reference_iclip(nfreq_ds, nt_ds, axis, &i_ds[0], &w_ds2[0], nt_ds, wrms.out_mean, wrms.out_rms, 1.0001 * sigma);

    // Step 4: upsample weights (twice)
    vector<float> w_ref1 = w_in;
    vector<float> w_ref2 = w_in;

    rf_kernels::weight_upsampler us(Df, Dt);
    us.upsample(nfreq_ds, nt_ds, &w_ref1[0], stride, &w_ds1[0], nt_ds);
    us.upsample(nfreq_ds, nt_ds, &w_ref2[0], stride, &w_ds2[0], nt_ds);

    // Compare!
    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt_chunk; it++) {
	    int i = ifreq * stride + it;
	    rf_assert(w_fast[i] <= w_ref1[i]);
	    rf_assert(w_fast[i] >= w_ref2[i]);
	}
    }
}


// -------------------------------------------------------------------------------------------------


static void test_wrms(std::mt19937 &rng, int niter_min, int niter_max)
{
    const int nouter = 1000;

    for (int iouter = 0; iouter < nouter; iouter++) {
	axis_type axis = AXIS_TIME;  // FIXME
	int Df = 1 << randint(rng, 0, 2);  // FIXME
	int Dt = 1 << randint(rng, 0, 2);  // FIXME
	int niter = randint(rng, niter_min, niter_max+1);
	int nfreq = Df * randint(rng, 1, 17);
	int nt = 8 * Dt * randint(rng, 1, 17);
	int stride = randint(rng, nt, 2*nt);
	double sigma = uniform_rand(rng, 1.5, 1.7);
	bool two_pass = true;  // FIXME

	test_wrms(rng, nfreq, nt, stride, axis, Df, Dt, niter, sigma, two_pass);
    }

    cout << "test_rms(niter_min=" << niter_min << ",niter_max=" << niter_max << "): pass" << endl;
}


static void test_intensity_clipper(std::mt19937 &rng, int niter_min, int niter_max)
{
    const int nouter = 1000;

    for (int iouter = 0; iouter < nouter; iouter++) {
	axis_type axis = AXIS_TIME;  // FIXME
	int Df = 1 << randint(rng, 0, 2);  // FIXME
	int Dt = 1 << randint(rng, 0, 2);  // FIXME
	int niter = randint(rng, niter_min, niter_max+1);
	int nfreq = Df * randint(rng, 1, 17);
	int nt = 8 * Dt * randint(rng, 1, 17);
	int stride = randint(rng, nt, 2*nt);
	double sigma = uniform_rand(rng, 1.0, 1.5);
	double iter_sigma = uniform_rand(rng, 1.5, 1.7);
	bool two_pass = true;  // FIXME

	test_intensity_clipper(rng, nfreq, nt, stride, axis, Df, Dt, niter, sigma, iter_sigma, two_pass);
    }

    cout << "test_intensity_clipper(niter_min=" << niter_min << ",niter_max=" << niter_max << "): pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_wrms(rng, 1, 1);
    test_intensity_clipper(rng, 1, 1);

    test_wrms(rng, 2, 2);
    test_intensity_clipper(rng, 2, 2);

    test_wrms(rng, 1, 4);
    test_intensity_clipper(rng, 1, 4);

    return 0;
}
