#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/std_dev_clipper.hpp"

using namespace std;
using namespace rf_kernels;


// Helper for reference_std_dev_clipper().
// The 'epsilon' parameter is used to exclude data when the variance
// is small compared to the (mean)^2, see below.

static void weighted_mean_variance_1d(float &out_mean, float &out_var, const float *intensity, const float *weights, int n, int stride, float epsilon=1.0e-10)
{
    // If we return early, then the (mean, variance) will be zero.
    out_mean = out_var = 0.0;

    float wsum = 0.0;
    float wisum = 0.0;
    
    // First pass
    for (int i = 0; i < n; i++) {
	wsum += weights[i*stride];
	wisum += weights[i*stride] * intensity[i*stride];
    }
	
    if (wsum <= 0.0)
	return;
	
    float mean = wisum / wsum;
    float wisum2 = 0.0;
    
    // Second pass
    for (int i = 0; i < n; i++)
	wisum2 += weights[i*stride] * square(intensity[i*stride] - mean);

    float var = wisum2 / wsum;
    if (var <= epsilon*mean*mean)
	return;

    out_mean = mean;
    out_var = var;
}


static void reference_std_dev_clipper(const float *intensity, float *weights, int nfreq, int nt, int stride, axis_type axis, double sigma, int Df, int Dt)
{
    // Not much argument checking!
    int nfreq_ds = xdiv(nfreq, Df);
    int nt_ds = xdiv(nt, Dt);

    float *intensity_ds = new float[nfreq_ds * nt_ds];
    float *weights_ds = new float[nfreq_ds * nt_ds];
	
    // Weighted downsample. (Some cut-and-paste with test-downsample.cpp here)
    
    for (int ifreq_d = 0; ifreq_d < nfreq_ds; ifreq_d++) {
	for (int it_d = 0; it_d < nt_ds; it_d++) {
	    float wisum = 0.0;
	    float wsum = 0.0;

	    for (int ifreq_u = ifreq_d*Df; ifreq_u < (ifreq_d+1)*Df; ifreq_u++) {
		for (int it_u = it_d*Dt; it_u < (it_d+1)*Dt; it_u++) {
		    int i = ifreq_u*stride + it_u;   // array index
		    wisum += weights[i] * intensity[i];
		    wsum += weights[i];
		}
	    }

	    intensity_ds[ifreq_d*nt_ds + it_d] = (wsum > 0.0) ? (wisum/wsum) : 0.0;
	    weights_ds[ifreq_d*nt_ds + it_d] = wsum;
	}
    }

    // Compute 1D variance.  This is done in a uniform way for AXIS_FREQ and AXIS_TIME,
    // by defining 'inner' and 'outer' variables.

    int ninner, nouter, istride, ostride;

    if (axis == AXIS_FREQ) {
	ninner = nfreq_ds;
	nouter = nt_ds;
	istride = nt_ds;
	ostride = 1;
    }
    else if (axis == AXIS_TIME) {
	ninner = nt_ds;
	nouter = nfreq_ds;
	istride = 1;
	ostride = nt_ds;
    }
    else
	throw runtime_error("bad axis in reference_std_dev_clipper");

    float *var_1d = new float[nouter];
    float *var_mask_1d = new float[nouter];

    for (int iouter = 0; iouter < nouter; iouter++) {
	float dummy_mean;
	weighted_mean_variance_1d(dummy_mean, var_1d[iouter], intensity_ds + iouter*ostride, weights_ds + iouter*ostride, ninner, istride);
	var_mask_1d[iouter] = (var_1d[iouter] > 0.0) ? 1.0 : 0.0;
    }

    // Compute the "variance of the variance"
    float var_mean, var_var;
    weighted_mean_variance_1d(var_mean, var_var, var_1d, var_mask_1d, nouter, 1);

    // Note: var_rms can be zero, if there are many failures!
    float var_rms = sqrt(var_var);

    // Mask the 1D variance array.
    for (int iouter = 0; iouter < nouter; iouter++) {
	// Use of ">=" (rather than ">") means that we always mask if var_rms=0.
	if (fabs(var_1d[iouter] - var_mean) >= sigma * var_rms)
	    var_mask_1d[iouter] = 0.0;
    }

    // Mask the downsampled array.

    for (int iouter = 0; iouter < nouter; iouter++) {
	if (var_mask_1d[iouter] > 0.0)
	    continue;
	for (int iinner = 0; iinner < ninner; iinner++)
	    weights_ds[iouter*ostride + iinner*istride] = 0.0;
    }

    // Mask the upsampled array.
    
    for (int ifreq_d = 0; ifreq_d < nfreq_ds; ifreq_d++) {
	for (int it_d = 0; it_d < nt_ds; it_d++) {
	    if (weights_ds[ifreq_d*nt_ds + it_d] > 0.0)
		continue;

	    for (int ifreq_u = ifreq_d*Df; ifreq_u < (ifreq_d+1)*Df; ifreq_u++)
		for (int it_u = it_d*Dt; it_u < (it_d+1)*Dt; it_u++)
		    weights[ifreq_u*stride + it_u] = 0.0;
	}
    }
	
    delete[] intensity_ds;
    delete[] weights_ds;
    delete[] var_1d;
    delete[] var_mask_1d;
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

    // Copy weights before running clippers.
    vector<float> in_w2 = in_w;
    vector<float> in_w3 = in_w;

    // Fast kernel.
    std_dev_clipper sd(nfreq, nt, axis, sigma, Df, Dt, two_pass);
    sd.clip(&in_i[0], &in_w[0], stride);

    // Reference kernels.
    reference_std_dev_clipper(&in_i[0], &in_w2[0], nfreq, nt, stride, axis, 0.999 * sigma, Df, Dt);
    reference_std_dev_clipper(&in_i[0], &in_w3[0], nfreq, nt, stride, axis, 1.001 * sigma, Df, Dt);

#if 0
    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt; it++) {
	    int i = ifreq * stride + it;
	    cout << "    (ifreq,it) = (" << ifreq << "," << it << "): w_fast=" << in_w[i] 
		 << ", w_ref1=" << in_w2[i] << ", w_ref2=" << in_w3[i] << endl;
	}
    }
#endif

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt; it++) {
	    int i = ifreq * stride + it;
	    rf_assert(in_w2[i] <= in_w[i]);
	    rf_assert(in_w3[i] >= in_w[i]);
	}
    }
}


static void test_std_dev_clipper(std::mt19937 &rng)
{
    for (int iter = 0; iter < 300; iter++) {
	int Df = 1 << randint(rng, 0, 6);
	int Dt = 1 << randint(rng, 0, 6);
	int nfreq = 8 * Df * randint(rng, 1, 17);  // FIXME
	int nt = 8 * Dt * randint(rng, 1, 17);
	int stride = randint(rng, nt, 2*nt);
	axis_type axis = randint(rng, 0, 2) ? AXIS_FREQ : AXIS_TIME;
	double sigma = uniform_rand(rng, 1.0, 1.5);
	bool two_pass = randint(rng, 0, 2);
	
	test_std_dev_clipper(rng, nfreq, nt, stride, axis, sigma, Df, Dt, two_pass);
    }

    cout << "test_std_dev_clipper: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    cout << "reminder: test-std-dev-clipper does not have complete generality yet" << endl;
    test_std_dev_clipper(rng);

    return 0;
}
