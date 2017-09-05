#include <simd_helpers/core.hpp>  // machine_epsilon()

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


// _ref_wrms_iterate(): 
//   - assumes (axis,Df,Dt) = (AXIS_NONE,1,1).
//   - assumes inout_mean has already been initialized.
//   - updates inout_mean, and fills out_rms.
//   - if the "epsilon" check fails, out_rms will be set to zero, but inout_mean will still be updated.

static void _ref_wrms_iterate(float *inout_mean, float *out_rms, int nfreq, int nt, const float *i_in, const float *w_in, int stride, float eps_multiplier)
{
    float wsum = 0.0;
    float wisum = 0.0;
    float wiisum = 0.0;
    float in_mean = *inout_mean;

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt; it++) {
	    float wval = w_in[ifreq*stride + it];
	    float dival = i_in[ifreq*stride + it] - in_mean;

	    wsum += wval;
	    wisum += wval * dival;
	    wiisum += wval * dival * dival;
	}
    }

    float dmean = (wsum > 0) ? (wisum/wsum) : 0;
    float var = (wsum > 0) ? (wiisum/wsum - dmean*dmean) : 0;

    float eps_2 = 1.0e2 * eps_multiplier * simd_helpers::machine_epsilon<float> ();
    float eps_3 = 1.0e3 * eps_multiplier * simd_helpers::machine_epsilon<float> ();

    // Threshold variance at (eps_2 in_mean)^2.
    // Note: use "<" here (not "<=") following rf_kernels/mean_rms_internals.hpp
    if (var < square(eps_2 * in_mean))
	var = 0.0;

    // Threshold variance at (eps_3 dmean^2).
    if (var < eps_3 * square(dmean))
	var = 0.0;

    *inout_mean = in_mean + dmean;
    *out_rms = sqrt(var);
}


// reference_wrms_iterate(): allows an arbitrary axis (otherwise identical to _ref_wrms_iterate).
//   - assumes (Df,Dt) = (1,1).
//   - assumes inout_mean has already been initialized.
//   - updates inout_mean, and fills out_rms.
//   - if the "epsilon" check fails, out_rms will be set to zero, but inout_mean will still be updated.

static void reference_wrms_iterate(float *inout_mean, float *out_rms, int nfreq, int nt, axis_type axis, const float *i_in, const float *w_in, int stride, float eps_multiplier)
{
    if (axis == AXIS_FREQ) {
	for (int it = 0; it < nt; it++)
	    _ref_wrms_iterate(inout_mean + it, out_rms + it, nfreq, 1, i_in + it, w_in + it, stride, eps_multiplier);
    }
    else if (axis == AXIS_TIME) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    _ref_wrms_iterate(inout_mean + ifreq, out_rms + ifreq, 1, nt, i_in + ifreq*stride, w_in + ifreq*stride, 0, eps_multiplier);
    }
    else if (axis == AXIS_NONE)
	_ref_wrms_iterate(inout_mean, out_rms, nfreq, nt, i_in, w_in, stride, eps_multiplier);
    else
	throw runtime_error("bad axis in reference_wrms()");
}


// reference_wrms_compute(): computes wrms from scratch, i.e. does not assume that 'out_mean' has been initialized.
static void reference_wrms_compute(float *out_mean, float *out_rms, int nfreq, int nt, axis_type axis, const float *i_in, const float *w_in, int stride, bool two_pass, float eps_multiplier)
{
    int nout = 1;
    if (axis == AXIS_FREQ) nout = nt;
    if (axis == AXIS_TIME) nout = nfreq;

    memset(out_mean, 0, nout * sizeof(*out_mean));
    
    reference_wrms_iterate(out_mean, out_rms, nfreq, nt, axis, i_in, w_in, stride, eps_multiplier);

    if (two_pass)
	reference_wrms_iterate(out_mean, out_rms, nfreq, nt, axis, i_in, w_in, stride, eps_multiplier);
}

// -------------------------------------------------------------------------------------------------


// Helper for reference_iclip: assumes (axis,Df,Dt)=(None,1,1) and (mean,thresh) already computed.
static void _ref_iclip(int nfreq, int nt, const float *i_in, float *w_in, int stride, float mean, float thresh)
{
    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt; it++) {
	    // Note: ">=" here (not ">"), so that we always mask if thresh=0.
	    if (abs(i_in[ifreq*stride+it] - mean) >= thresh)
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


static void _simulate_evil_data(std::mt19937 &rng, float *intensity, float *weights, int n, bool permute)
{
    float wsum = 0.0;
    float wisum = 0.0;
    float wiisum = 0.0;

    // Give simulated data a 15% chance of being "sparse".
    float pnonzero = (uniform_rand(rng) < 0.1) ? (1.5/n) : 1.0;

    // Simulate data.
    for (int i = 0; i < n; i++) {
	intensity[i] = uniform_rand(rng, -1.0, 1.0);
	weights[i] = (uniform_rand(rng) < pnonzero) ? uniform_rand(rng) : 0.0;

	wsum += weights[i];
	wisum += weights[i] * intensity[i];
    }

    if (wsum == 0.0)
	return;

    // Now normalize simulated data to (mean,rms)=(0,1),
    // so that we can set the (mean,rms) to specified values next.

    float mean = wisum / wsum;

    for (int i = 0; i < n; i++) {
	intensity[i] -= mean;
	wiisum += weights[i] * square(intensity[i]);
    }

    if (wiisum == 0.0)
	return;

    float rms = sqrt(wiisum/wsum);

    for (int i = 0; i < n; i++)
	intensity[i] /= rms;

    // Decide what we want the (mean,rms) to be.

    mean = uniform_rand(rng, -1.0, 1.0);
    rms = uniform_rand(rng);

    // Assign specified (mean,rms) to the simulated data.

    for (int i = 0; i < n; i++)
	intensity[i] = rms * intensity[i] + mean;
}


static void simulate_evil_data(std::mt19937 &rng, int nfreq, int nt, float *intensity, float *weights, int stride, axis_type axis)
{
    if (axis == AXIS_FREQ) {
	vector<float> i_tmp(nfreq, 0.0);
	vector<float> w_tmp(nfreq, 0.0);

	for (int it = 0; it < nt; it++) {
	    _simulate_evil_data(rng, &i_tmp[0], &w_tmp[0], nfreq, true);

	    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
		intensity[ifreq*stride+it] = i_tmp[ifreq];
		weights[ifreq*stride+it] = w_tmp[ifreq];
	    }
	}
    }
    else if (axis == AXIS_TIME) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    _simulate_evil_data(rng, intensity + ifreq*stride, weights + ifreq*stride, nt, true);
    }
    else if (axis == AXIS_NONE) {
	vector<float> i_tmp(nfreq * nt, 0.0);
	vector<float> w_tmp(nfreq * nt, 0.0);
	_simulate_evil_data(rng, &i_tmp[0], &w_tmp[0], nfreq*nt, true);
	
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    for (int it = 0; it < nt; it++) {
		intensity[ifreq*stride+it] = i_tmp[ifreq*nt+it];
		weights[ifreq*stride+it] = w_tmp[ifreq*nt+it];
	    }
	}
    }
    else
	throw runtime_error("bad axis in simulate()");
}


// -------------------------------------------------------------------------------------------------


static void test_wrms(std::mt19937 &rng, int nfreq, int nt_chunk, int stride, axis_type axis, int Df, int Dt, int niter, double sigma, bool two_pass)
{
    int verbosity = 0;  // increase for debugging
    int nfreq_ds = xdiv(nfreq, Df);
    int nt_ds = xdiv(nt_chunk, Dt);
    
    int nout = 1;
    if (axis == AXIS_FREQ) nout = nt_ds;
    if (axis == AXIS_TIME) nout = nfreq_ds;

    vector<float> i_in(nfreq*stride, 0.0);
    vector<float> w_in(nfreq*stride, 0.0);
    simulate_evil_data(rng, nfreq, nt_chunk, &i_in[0], &w_in[0], stride, axis);

    rf_kernels::weighted_mean_rms wrms(nfreq, nt_chunk, axis, Df, Dt, niter, sigma, two_pass);    
    rf_assert(wrms.nfreq_ds == nfreq_ds);
    rf_assert(wrms.nt_ds == nt_ds);
    rf_assert(wrms.nout == nout);

    // Run fast wrms kernel.
    // Reminder: the outputs are stored in wrms.out_mean[] and wrms.out_rms[].
    wrms.compute_wrms(&i_in[0], &w_in[0], stride);

    // The rest of this routine is devoted to computing something to compare to!

    // Outputs from reference wrms will go here.  We call reference_wrms_compute() twice,
    // with different epsilon_multipliers, in order to make the test roundoff-robust.

    vector<float> refc_mean(nout, 0.0);   // refc = "conservative" (eps_multiplier=1.5)
    vector<float> refc_rms(nout, 0.0);
    vector<float> refp_mean(nout, 0.0);   // refp = "permissive" (eps_multiplier=0.5)
    vector<float> refp_rms(nout, 0.0);

    // Temporary buffers.
    vector<float> i_ds(nfreq_ds * nt_ds, 0.0);
    vector<float> w_ds(nfreq_ds * nt_ds, 0.0);

    // This will store the initial mean "hint" used in the last iteration of the wrms kernel.
    // This is needed to assign statistical errors to the output wrms.
    vector<float> mean_hint(nout, 0.0);

    if (two_pass)
	memcpy(&mean_hint[0], wrms.out_mean, nout * sizeof(float));
	
    // Step 1: apply reference kernel.
    // Warning: we may end up overwriting the 'w_in' array (with an intensity clipper),
    // but that's OK, we won't need it in step 2.

    if (niter == 1) {
	rf_kernels::wi_downsampler ds(Df, Dt);
	ds.downsample(nfreq_ds, nt_ds, &i_ds[0], &w_ds[0], nt_ds, &i_in[0], &w_in[0], stride);

	reference_wrms_compute(&refc_mean[0], &refc_rms[0], nfreq_ds, nt_ds, axis, &i_ds[0], &w_ds[0], nt_ds, two_pass, 1.5);
	reference_wrms_compute(&refp_mean[0], &refp_rms[0], nfreq_ds, nt_ds, axis, &i_ds[0], &w_ds[0], nt_ds, two_pass, 0.5);
    }
    else {
	// The following logic is convenient but does a little redundant computation!
	// Call the fast wrms kernel with (N-1) iterations, to initialize the 'mean_hint' array needed by reference_wrms_iterate().
	rf_kernels::weighted_mean_rms wrms2(nfreq, nt_chunk, axis, Df, Dt, niter-1, sigma, two_pass);
	wrms2.compute_wrms(&i_in[0], &w_in[0], stride);

	memcpy(&refc_mean[0], wrms2.out_mean, nout * sizeof(float));
	memcpy(&refp_mean[0], wrms2.out_mean, nout * sizeof(float));
	memcpy(&mean_hint[0], wrms2.out_mean, nout * sizeof(float));
	
	// Account for the first (niter-1) iterations by clipping.
	rf_kernels::intensity_clipper ic(nfreq, nt_chunk, axis, sigma, Df, Dt, niter-1, 0, two_pass);
	ic.clip(&i_in[0], &w_in[0], stride);

	rf_kernels::wi_downsampler ds(Df, Dt);
	ds.downsample(nfreq_ds, nt_ds, &i_ds[0], &w_ds[0], nt_ds, &i_in[0], &w_in[0], stride);

	reference_wrms_iterate(&refc_mean[0], &refc_rms[0], nfreq_ds, nt_ds, axis, &i_ds[0], &w_ds[0], nt_ds, 1.5);
	reference_wrms_iterate(&refp_mean[0], &refp_rms[0], nfreq_ds, nt_ds, axis, &i_ds[0], &w_ds[0], nt_ds, 0.5);
    }

    // Step 2: compare outputs!    

    // First we decide what statistical errors are allowed on the mean and rms.

    vector<float> eps_m(nout, 0.0);
    vector<float> eps_r(nout, 0.0);

    for (int i = 0; i < nout; i++) {
	float r = wrms.out_rms[i];
	float m = abs(mean_hint[i]) + abs(wrms.out_mean[i]);
	float dm = abs(mean_hint[i] - wrms.out_mean[i]);

	eps_m[i] = (1.0e-4 * m) + (1.0e-4 * r);
	eps_r[i] = (1.0e-2 * sqrt(dm)) + (1.0e-4 * m) + (1.0e-4 * r);
    }

    // Now the comparison..

    int icmp = 0;

    try {
	while (icmp < nout) {
	    if (wrms.out_rms[icmp] > 0) {
		rf_assert(refp_rms[icmp] > 0);
		rf_assert(abs(wrms.out_rms[icmp] - refp_rms[icmp]) <= eps_m[icmp]);
		rf_assert(abs(wrms.out_mean[icmp] - refp_mean[icmp]) <= eps_r[icmp]);
	    }
	    else
		rf_assert(refc_rms[icmp] == 0);

	    icmp++;
	}
    } catch (exception &e) {
	cout << e.what() << endl;
	// fall through...
    }

    bool failed = (icmp < nout);

    if (failed || (verbosity >= 1)) {
	cout << "test_wrms(nfreq=" << nfreq << ",nt_chunk=" << nt_chunk << ",stride=" << stride << ",axis=" << axis
	     << ",Df=" << Df << ",Dt=" << Dt << ",niter=" << niter << ",sigma=" << sigma << ",two_pass=" << two_pass << ")\n";
    }

    if (failed) {
	cout << "    failed at i=" << icmp << endl
	     << "    fast: mean=" << wrms.out_mean[icmp] << ", rms=" << wrms.out_rms[icmp] << endl
	     << "    refc: mean=" << refc_mean[icmp] << ", rms=" << refc_rms[icmp] << endl
	     << "    refp: mean=" << refp_mean[icmp] << ", rms=" << refp_rms[icmp] << endl
	     << "    eps_m=" << eps_m[icmp] << ", eps_r=" << eps_r[icmp] << ", mean_hint=" << mean_hint[icmp] << endl;

	exit(1);
    }
}


// -------------------------------------------------------------------------------------------------


static void test_intensity_clipper(std::mt19937 &rng, int nfreq, int nt_chunk, int stride, axis_type axis, int Df, int Dt, int niter, double sigma, double iter_sigma, bool two_pass)
{
    int verbosity = 0;  // increase for debugging
    int nfreq_ds = xdiv(nfreq, Df);
    int nt_ds = xdiv(nt_chunk, Dt);
    
    vector<float> i_in(nfreq*stride, 0.0);
    vector<float> w_in(nfreq*stride, 0.0);
    simulate_evil_data(rng, nfreq, nt_chunk, &i_in[0], &w_in[0], stride, axis);

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
    wrms.compute_wrms(&i_ds[0], &w_ds[0], nt_ds);

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

    // Step 5: Compare!

    int ifreq_c = 0;
    int it_c = 0;

    try {
	while (ifreq_c < nfreq) {
	    while (it_c < nt_chunk) {
		int i = ifreq_c * stride + it_c;
		rf_assert(w_fast[i] >= w_ref1[i]);
		rf_assert(w_fast[i] <= w_ref2[i]);
		it_c++;
	    }
	    ifreq_c++;
	}
    }
    catch (exception &e) {
	cout << e.what() << endl;
	// fall through...
    }

    bool failed = (ifreq_c < nfreq) || (it_c < nt_chunk);

    if (failed || (verbosity >= 1)) {
	cout << "test_intensity_clipper(nfreq=" << nfreq << ",nt_chunk=" << nt_chunk << ",stride=" << stride 
	     << ",axis=" << axis << ",Df=" << Df << ",Dt=" << Dt << ",niter=" << niter << ",sigma=" << sigma 
	     << ",iter_sigma=" << iter_sigma << ",two_pass=" << two_pass << ")" << endl;
    }

    if (failed) {
	cout << "    failed at ifreq=" << ifreq_c << ", it=" << it_c << endl;

	if (axis == AXIS_FREQ)
	    cout << "    at it=" << it_c << ", mean=" << wrms.out_mean[it_c] << ", rms=" << wrms.out_rms[it_c] << endl;
	if (axis == AXIS_TIME)	
	    cout << "    at ifreq=" << ifreq_c << ", mean=" << wrms.out_mean[ifreq_c] << ", rms=" << wrms.out_rms[ifreq_c] << endl;
	if (axis == AXIS_NONE)
	    cout << "    global mean=" << wrms.out_mean[0] << ", rms=" << wrms.out_rms[0] << endl;

	int i = ifreq_c * stride + it_c;
	cout << "    at (ifreq,it) = (" << ifreq_c << "," << it_c << "): i_in= " << i_in[i]
	     << ", w_fast=" << w_fast[i] << ", w_ref1=" << w_ref1[i] << ", w_ref2=" << w_ref2[i] << endl;

	exit(1);
    }
}


// -------------------------------------------------------------------------------------------------


static void test_wrms(std::mt19937 &rng, int niter_min, int niter_max)
{
    cout << "test_rms(niter_min=" << niter_min << ",niter_max=" << niter_max << "): start" << endl;

    for (int iouter = 0; iouter < 300; iouter++) {
	axis_type axis = random_axis_type(rng);
	int Df = 1 << randint(rng, 0, 6);
	int Dt = 1 << randint(rng, 0, 6);
	int niter = randint(rng, niter_min, niter_max+1);
	int nfreq = Df * randint(rng, 1, 65);
	int nt = 8 * Dt * randint(rng, 1, 17);
	int stride = randint(rng, nt, 2*nt);
	double sigma = uniform_rand(rng, 1.5, 1.7);
	bool two_pass = randint(rng, 0, 2);

	test_wrms(rng, nfreq, nt, stride, axis, Df, Dt, niter, sigma, two_pass);
    }
}


static void test_intensity_clipper(std::mt19937 &rng, int niter_min, int niter_max)
{
    cout << "test_intensity_clipper(niter_min=" << niter_min << ",niter_max=" << niter_max << "): start" << endl;

    for (int iouter = 0; iouter < 300; iouter++) {
	axis_type axis = random_axis_type(rng);
	int Df = 1 << randint(rng, 0, 6);
	int Dt = 1 << randint(rng, 0, 6);
	int niter = randint(rng, niter_min, niter_max+1);
	int nfreq = Df * randint(rng, 1, 65);
	int nt = 8 * Dt * randint(rng, 1, 17);
        int stride = randint(rng, nt, 2*nt);
	double sigma = uniform_rand(rng, 1.0, 1.5);
	double iter_sigma = uniform_rand(rng, 1.5, 1.7);
	bool two_pass = randint(rng, 0, 2);

	test_intensity_clipper(rng, nfreq, nt, stride, axis, Df, Dt, niter, sigma, iter_sigma, two_pass);
    }
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

    test_wrms(rng, 1, 5);
    test_intensity_clipper(rng, 1, 5);

    cout << "test-intensity-clipper: all tests passed" << endl;
    return 0;
}
