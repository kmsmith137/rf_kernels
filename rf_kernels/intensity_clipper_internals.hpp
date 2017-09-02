#ifndef _RF_KERNELS_INTENSITY_CLIPPER_INTERNALS_HPP
#define _RF_KERNELS_INTENSITY_CLIPPER_INTERNALS_HPP

#include "upsample_internals.hpp"
#include "downsample_internals.hpp"
#include "mean_rms_internals.hpp"
#include "clipper_internals.hpp"
#include "intensity_clipper.hpp"

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif

template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;


// -------------------------------------------------------------------------------------------------
//
// masking kernels
//
// If the intensity differs from the mean by more than 'thresh', set weights to zero.
//
// The intensity array can be downsampled relative to the weights!


template<typename T, int S, int Df, int Dt>
inline void _kernel_intensity_mask_2d(T *weights, const T *ds_intensity, simd_t<T,S> mean, simd_t<T,S> thresh, int nfreq, int nt, int stride, int ds_stride)
{
    simd_t<T,S> lo = mean - thresh;
    simd_t<T,S> hi = mean + thresh;

    const T *ds_irow = ds_intensity;

    for (int ifreq = 0; ifreq < nfreq; ifreq += Df) {
	const T *ds_itmp = ds_irow;
	T *wrow = weights + ifreq * stride;

	for (int it = 0; it < nt; it += Dt*S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S>(ds_itmp);
	    ds_itmp += S;

	    smask_t<T,S> valid1 = ival.compare_lt(hi);
	    smask_t<T,S> valid2 = ival.compare_gt(lo);
	    smask_t<T,S> valid = valid1.bitwise_and(valid2);

	    _kernel_mask<T,S,Df,Dt> (wrow + it, valid, stride);
	}

	ds_irow += ds_stride;
    }
}


template<typename T, int S, int Df, int Dt>
inline void _kernel_intensity_mask_1d_t(T *weights, const T *ds_intensity, simd_t<T,S> mean, simd_t<T,S> thresh, int nt, int stride, int ds_stride)
{
    _kernel_intensity_mask_2d<T,S,Df,Dt> (weights, ds_intensity, mean, thresh, Df, nt, stride, ds_stride);
}


template<typename T, int S, int Df, int Dt>
inline void _kernel_intensity_mask_1d_f(T *weights, const T *ds_intensity, simd_t<T,S> mean, simd_t<T,S> thresh, int nfreq, int stride, int ds_stride)
{
    simd_t<T,S> lo = mean - thresh;
    simd_t<T,S> hi = mean + thresh;

    for (int ifreq = 0; ifreq < nfreq; ifreq += Df) {
	simd_t<T,S> ival = simd_helpers::simd_load<T,S> (ds_intensity);
	ds_intensity += ds_stride;

	smask_t<T,S> valid1 = ival.compare_lt(hi);
	smask_t<T,S> valid2 = ival.compare_gt(lo);
	smask_t<T,S> valid = valid1.bitwise_and(valid2);

	_kernel_mask<T,S,Df,Dt> (weights + ifreq*stride, valid, stride);
    }
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_clip_2d(): "Bottom line" routine which is wrapped by intensity_clipper(AXIS_NONE).


// Downsampled version: ds_intensity must be non-NULL, ds_weights must be non-NULL if niter > 1.
template<typename T, int S, int Df, int Dt, bool TwoPass, typename std::enable_if<((Df>1) || (Dt>1)),int>::type = 0>
inline void _kernel_clip_2d(const T *intensity, T *weights, int nfreq, int nt, int stride, int niter, double sigma, double iter_sigma, T *ds_intensity, T *ds_weights)
{
    simd_t<T,S> mean, rms;

    if (niter == 1)
	_kernel_noniterative_wrms_2d<T,S,Df,Dt,true,false,TwoPass> (mean, rms, intensity, weights, nfreq, nt, stride, ds_intensity, ds_weights);
    else {
	_kernel_noniterative_wrms_2d<T,S,Df,Dt,true,true,TwoPass> (mean, rms, intensity, weights, nfreq, nt, stride, ds_intensity, ds_weights);
	_kernel_wrms_iterate_2d<T,S> (mean, rms, ds_intensity, ds_weights, nfreq/Df, nt/Dt, nt/Dt, niter, iter_sigma);
    }

    simd_t<T,S> thresh = simd_t<T,S>(sigma) * rms;
    _kernel_intensity_mask_2d<T,S,Df,Dt> (weights, ds_intensity, mean, thresh, nfreq, nt, stride, nt/Dt);
}

// Non-downsampled version: ds_intensity, ds_weights can be NULL
template<typename T, int S, int Df, int Dt, bool TwoPass, typename std::enable_if<((Df==1) && (Dt==1)),int>::type = 0>
inline void _kernel_clip_2d(const T *intensity, T *weights, int nfreq, int nt, int stride, int niter, double sigma, double iter_sigma, T *ds_intensity, T *ds_weights)
{
    simd_t<T,S> mean, rms;
    _kernel_noniterative_wrms_2d<T,S,1,1,false,false,TwoPass> (mean, rms, intensity, weights, nfreq, nt, stride, NULL, NULL);
    _kernel_wrms_iterate_2d<T,S> (mean, rms, intensity, weights, nfreq, nt, stride, niter, iter_sigma);

    simd_t<T,S> thresh = simd_t<T,S>(sigma) * rms;
    _kernel_intensity_mask_2d<T,S,Df,Dt> (weights, intensity, mean, thresh, nfreq, nt, stride, stride);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_clip_1d_t(): "Bottom line" routine which is wrapped by intensity_clipper(AXIS_TIME).


template<typename T, int S, int Df, int Dt, bool TwoPass, typename std::enable_if<((Df>1) || (Dt>1)),int>::type = 0>
inline void _kernel_clip_1d_t(const T *intensity, T *weights, int nfreq, int nt, int stride, int niter, double sigma, double iter_sigma, T *ds_intensity, T *ds_weights)
{
    simd_t<T,S> mean, rms;
    simd_t<T,S> s = sigma;

    if (niter > 1) {
	for (int ifreq = 0; ifreq < nfreq; ifreq += Df) {
	    _kernel_noniterative_wrms_1d_t<T,S,Df,Dt,true,true,TwoPass> (mean, rms, intensity + ifreq*stride, weights + ifreq*stride, nt, stride, ds_intensity, ds_weights);
	    _kernel_wrms_iterate_1d_t<T,S> (mean, rms, ds_intensity, ds_weights, nt/Dt, niter, iter_sigma);
	    _kernel_intensity_mask_1d_t<T,S,Df,Dt> (weights + ifreq*stride, ds_intensity, mean, s * rms, nt, stride, nt/Dt);
	}
    }
    else {
	for (int ifreq = 0; ifreq < nfreq; ifreq += Df) {
	    _kernel_noniterative_wrms_1d_t<T,S,Df,Dt,true,false,TwoPass> (mean, rms, intensity + ifreq*stride, weights + ifreq*stride, nt, stride, ds_intensity, ds_weights);
	    _kernel_intensity_mask_1d_t<T,S,Df,Dt> (weights + ifreq*stride, ds_intensity, mean, s * rms, nt, stride, nt/Dt);
	}
    }
}


template<typename T, int S, int Df, int Dt, bool TwoPass, typename std::enable_if<((Df==1) && (Dt==1)),int>::type = 0>
inline void _kernel_clip_1d_t(const T *intensity, T *weights, int nfreq, int nt, int stride, int niter, double sigma, double iter_sigma, T *ds_intensity, T *ds_weights)
{
    simd_t<T,S> mean, rms;
    simd_t<T,S> s = sigma;

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	const T *irow = intensity + ifreq * stride;
	T *wrow = weights + ifreq * stride;

	_kernel_noniterative_wrms_1d_t<T,S,1,1,false,false,TwoPass> (mean, rms, irow, wrow, nt, stride, NULL, NULL);
	_kernel_wrms_iterate_1d_t<T,S> (mean, rms, irow, wrow, nt, niter, iter_sigma);
	_kernel_intensity_mask_1d_t<T,S,1,1> (wrow, irow, mean, s * rms, nt, stride, stride);
    }
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_clip_1d_f(): "Bottom line" routine which is wrapped by intensity_clipper(AXIS_FREQ).


template<typename T, int S, int Df, int Dt, bool TwoPass, typename std::enable_if<((Df > 1) || (Dt > 1)),int>::type = 0>
inline void _kernel_clip_1d_f(const T *intensity, T *weights, int nfreq, int nt, int stride, int niter, double sigma, double iter_sigma, T *ds_intensity, T *ds_weights)
{
    simd_t<T,S> mean, rms;	
    simd_t<T,S> s = sigma;

    if (niter > 1) {
	for (int it = 0; it < nt; it += Dt*S) {
	    _kernel_noniterative_wrms_1d_f<T,S,Df,Dt,true,true,TwoPass> (mean, rms, intensity + it, weights + it, nfreq, stride, ds_intensity, ds_weights);
	    _kernel_wrms_iterate_1d_f<T,S> (mean, rms, ds_intensity, ds_weights, nfreq/Df, S, niter, iter_sigma);
	    _kernel_intensity_mask_1d_f<T,S,Df,Dt> (weights + it, ds_intensity, mean, s * rms, nfreq, stride, S);
	}
    }
    else {
	for (int it = 0; it < nt; it += Dt*S) {
	    _kernel_noniterative_wrms_1d_f<T,S,Df,Dt,true,false,TwoPass> (mean, rms, intensity + it, weights + it, nfreq, stride, ds_intensity, ds_weights);
	    _kernel_intensity_mask_1d_f<T,S,Df,Dt> (weights + it, ds_intensity, mean, s * rms, nfreq, stride, S);
	}
    }
}


template<typename T, int S, int Df, int Dt, bool TwoPass, typename std::enable_if<((Df == 1) && (Dt == 1)),int>::type = 0>
inline void _kernel_clip_1d_f(const T *intensity, T *weights, int nfreq, int nt, int stride, int niter, double sigma, double iter_sigma, T *ds_intensity, T *ds_weights)
{
    simd_t<T,S> mean, rms;	
    simd_t<T,S> s = sigma;

    for (int it = 0; it < nt; it += S) {
	const T *icol = intensity + it;
	T *wcol = weights + it;
	
	_kernel_noniterative_wrms_1d_f<T,S,1,1,false,false,TwoPass> (mean, rms, icol, wcol, nfreq, stride, NULL, NULL);
	_kernel_wrms_iterate_1d_f<T,S> (mean, rms, icol, wcol, nfreq, stride, niter, iter_sigma);
	_kernel_intensity_mask_1d_f<T,S,1,1> (wcol, icol, mean, s * rms, nfreq, stride, stride);
    }
}


// =================================================================================================
// =================================================================================================
// =================================================================================================
// =================================================================================================



template<typename T, int S, int Df, int Dt>
inline void kernel_iclip_Dfsm_Dtsm(const intensity_clipper *ic, const T *in_i, T *in_w, int stride)
{
    int nfreq_ds = ic->nfreq_ds;
    int nt_ds = ic->nt_ds;
    int niter = ic->niter;
    float *tmp_i = ic->tmp_i;
    float *tmp_w = ic->tmp_w;
    simd_t<T,S> sigma(ic->sigma);
    simd_t<T,S> iter_sigma(ic->iter_sigma);
    
    _wi_downsampler_0d_Dtsm<T,S,Df,Dt> ds0;
    _wi_downsampler_1d_Dfsm<decltype(ds0)> ds1(ds0);
    _weight_upsampler_0d_Dtsm<T,S,Df,Dt> us0;

    for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++) {
	const T *in_i2 = in_i + ifreq_ds * Df * stride;
	T *in_w2 = in_w + ifreq_ds * Df * stride;

	_wrms_1d_outbuf<T,S> out(tmp_i, tmp_w, nt_ds, iter_sigma);
	
	ds1.downsample_1d(out, nt_ds, in_i2, in_w2, stride);
	out.end_row();

	// (niter-1) iterations
	for (int iter = 1; iter < niter; iter++)
	    out.iterate();
	
	for (int it = 0; it < nt_ds; it += S) {
	    simd_t<T,S> mask = out.get_mask(it);
	    us0.put_mask(in_w2 + it*Dt, stride, mask);
	}
    }
}



}  // namespace rf_kernels

#endif // _RF_KERNELS_INTENSITY_CLIPPER_INTERNALS_HPP
