#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/mean_rms_internals.hpp"
#include "rf_kernels/mean_rms.hpp"

using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


weighted_mean_rms::weighted_mean_rms(int nfreq_, int nt_chunk_, int niter_, double sigma_, bool two_pass_) :
    nfreq(nfreq_),
    nt_chunk(nt_chunk_),
    niter(niter_),
    sigma(sigma_),
    two_pass(two_pass_)
{ 
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::intensity_clipper: expected nfreq > 0");

    if (_unlikely(nt_chunk <= 0))
	throw runtime_error("rf_kernels::intensity_clipper: expected nt_chunk > 0");
    
    if (_unlikely((nt_chunk % 8) != 0))
	throw runtime_error("rf_kernels::intensity_clipper: expected nt_chunk to be a multiple of 8");

    if (_unlikely(sigma < 1.0))
	throw runtime_error("rf_kernels::intensity_clipper: expected sigma >= 1.0");

    if (_unlikely(niter < 1))
	throw runtime_error("rf_kernels::intensity_clipper: expected niter >= 1");
}


template<typename T, int S>
inline void _weighted_mean_and_rms(simd_t<T,S> &mean, simd_t<T,S> &rms, const float *intensity, const float *weights, int nfreq, int nt, int stride, int niter, double sigma, bool two_pass)
{
    if (two_pass)
	_kernel_noniterative_wrms_2d<T,S,1,1,false,false,true> (mean, rms, intensity, weights, nfreq, nt, stride, NULL, NULL);
    else
	_kernel_noniterative_wrms_2d<T,S,1,1,false,false,false> (mean, rms, intensity, weights, nfreq, nt, stride, NULL, NULL);

    _kernel_wrms_iterate_2d<T,S> (mean, rms, intensity, weights, nfreq, nt, stride, niter, sigma);
}


void weighted_mean_rms::compute_wrms(float &mean, float &rms, const float *intensity, const float *weights, int stride)
{
    constexpr int S = 8;
    
    if (_unlikely(!intensity || !weights))
	throw runtime_error("rf_kernels: null pointer passed to weighted_mean_rms::compute_wrms()");

    if (_unlikely(abs(stride) < nt_chunk))
	throw runtime_error("rf_kernels::weighed_mean_rms: stride is too small");

    simd_t<float,S> mean_x, rms_x;
    _weighted_mean_and_rms(mean_x, rms_x, intensity, weights, nfreq, nt_chunk, stride, niter, sigma, two_pass);

    rms = rms_x.template extract<0> ();
    mean = (rms > 0.0) ? (mean_x.template extract<0> ()) : 0.0;
}


}  // namespace rf_kernels
