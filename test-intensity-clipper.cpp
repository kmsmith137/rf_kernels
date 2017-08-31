#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/intensity_clipper.hpp"

using namespace std;
using namespace rf_kernels;


// -------------------------------------------------------------------------------------------------


// Helper for reference_intensity_clipper().
// The 'epsilon' parameter is used to exclude data when the variance
// is small compared to the (mean)^2, see below.

static void weighted_mean_and_rms(float &out_mean, float &out_rms, const float *intensity, const float *weights, int nfreq, int nt, int stride, float epsilon=1.0e-10)
{
    // If we return early, then the (mean, rms) will be zero.
    out_mean = out_rms = 0.0;
    
    float wsum = 0.0;
    float wisum = 0.0;
    
    // First pass
    for (int i = 0; i < nfreq; i++) {
	for (int j = 0; j < nt; j++) {
	    wsum += weights[i*stride+j];
	    wisum += weights[i*stride+j] * intensity[i*stride+j];
	}
    }
	
    if (wsum <= 0.0)
	return;
	
    float mu = wisum / wsum;
    float wisum2 = 0.0;
	
    // Second pass
    for (int i = 0; i < nfreq; i++)
	for (int j = 0; j < nt; j++)
	    wisum2 += weights[i*stride+j] * square(intensity[i*stride+j] - mu);

    float var = wisum2 / wsum;
    if (var <= epsilon*mu*mu)
	return;

    out_mean = mu;
    out_rms = sqrt(var);
}


// Non-iterative for now!
static void reference_intensity_clipper(const float *intensity, float *weights, int nfreq, int nt, int stride, axis_type axis, double sigma, int Df, int Dt)
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

    // Compute weighted mean/rms of downsampled array.
    //
    // We do this in a uniform way for all 'axis' values, by interpreting the
    // output as an array whose shape (nfreq_mr, nt_mr) is:
    //   
    //   (1, nt_ds)     if axis==AXIS_FREQ
    //   (nfreq_ds, 1)  if axis==AXIS_TIME
    //   (1, 1)         if axis==AXIS_NONE
    
    int nfreq_mr = (axis == AXIS_TIME) ? nfreq_ds : 1;
    int nt_mr = (axis == AXIS_FREQ) ? nt_ds : 1;
    int Mf = xdiv(nfreq_ds, nfreq_mr);
    int Mt = xdiv(nt_ds, nt_mr);

    float *mean_arr = new float[nfreq_mr * nt_mr];
    float *rms_arr = new float[nfreq_mr * nt_mr];
    
    for (int ifreq_m = 0; ifreq_m < nfreq_mr; ifreq_m++) {
	for (int it_m = 0; it_m < nt_mr; it_m++) {
	    weighted_mean_and_rms(mean_arr[ifreq_m*nt_mr + it_m],              // out_mean
				  rms_arr[ifreq_m*nt_mr + it_m],               // out_rms
				  &intensity_ds[ifreq_m*Mf*nt_ds + it_m*Mt],   // intensity
				  &weights_ds[ifreq_m*Mf*nt_ds + it_m*Mt],     // weights
				  Mf, Mt, nt_ds);                              // nfreq, nt, stride
	}
    }

    // Threshold the downsampled array.

    for (int ifreq_m = 0; ifreq_m < nfreq_mr; ifreq_m++) {
	for (int it_m = 0; it_m < nt_mr; it_m++) {
	    // Note: rms can be zero, if weighted_mean_and_rms() failed.
	    float mean = mean_arr[ifreq_m*nt_mr + it_m];
	    float rms = rms_arr[ifreq_m*nt_mr + it_m];

	    for (int ifreq_d = ifreq_m*Mf; ifreq_d < (ifreq_m+1)*Mf; ifreq_d++) {
		for (int it_d = it_m*Mt; it_d < (it_m+1)*Mt; it_d++) {
		    // Index in downsampled array.
		    int i = ifreq_d*nt_ds + it_d;
			
		    // Use of ">=" (rather than ">") means that we always mask if rms=0.
		    if (abs(intensity_ds[i]-mean) >= sigma * rms)
			weights_ds[i] = 0.0;
		}
	    }
	}
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
    delete[] mean_arr;
    delete[] rms_arr;
}


static void test_intensity_clipper(std::mt19937 &rng, int nfreq, int nt, int stride, axis_type axis, double sigma, int Df, int Dt, int niter)
{
#if 0
    cout << "test_intensity_clipper: nfreq=" << nfreq << ", nt=" << nt 
	 << ", stride=" << stride << ", axis=" << axis << ", sigma=" << sigma 
	 << ", Df=" << Df << ", Dt=" << Dt << ", niter=" << niter << endl;
#endif

    vector<float> in_i = uniform_randvec(rng, nfreq * stride, 0.0, 1.0);
    vector<float> in_w = vector<float> (nfreq * stride, 0.0);

    // Low value chosen to expose corner cases.
    float pnonzero = 1.0 / float(Df*Dt);
    for (size_t i = 0; i < in_w.size(); i++)
	in_w[i] = (uniform_rand(rng) < pnonzero) ? uniform_rand(rng) : 0.0;

    // Copy weights before running clippers.
    vector<float> in_w2 = in_w;

    // Fast kernel
    intensity_clipper ic(nfreq, nt, axis, sigma, Df, Dt, niter);
    ic.clip(&in_i[0], &in_w[0], stride);

    // Fast kernel (niter-1)
    if (niter > 1) {
	intensity_clipper ic2(nfreq, nt, axis, sigma, Df, Dt, niter-1);
	ic2.clip(&in_i[0], &in_w2[0], stride);
    }

    // Reference kernels
    vector<float> in_w3 = in_w2;
    reference_intensity_clipper(&in_i[0], &in_w2[0], nfreq, nt, stride, axis, 0.999 * sigma, Df, Dt);
    reference_intensity_clipper(&in_i[0], &in_w3[0], nfreq, nt, stride, axis, 1.001 * sigma, Df, Dt);

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int it = 0; it < nt; it++) {
	    int i = ifreq * stride + it;
	    rf_assert(in_w2[i] <= in_w[i]);
	    rf_assert(in_w3[i] >= in_w[i]);
	}
    }
}


static void test_intensity_clipper(std::mt19937 &rng)
{
    for (int iter = 0; iter < 1000; iter++) {
	int Df = 1 << randint(rng, 0, 4);
	int Dt = 1 << randint(rng, 0, 4);
	int nfreq = Df * randint(rng, 1, 17);
	int nt = 8 * Dt * randint(rng, 1, 17);
	int stride = randint(rng, nt, 2*nt);
	double sigma = uniform_rand(rng, 1.0, 1.5);
	axis_type axis = random_axis_type(rng);
	int niter = 1;   // FIXME
	
	test_intensity_clipper(rng, nfreq, nt, stride, axis, sigma, Df, Dt, niter);
    }

    cout << "test_intensity_clipper: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_intensity_clipper(rng);

    return 0;
}
