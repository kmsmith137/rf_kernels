#ifndef _RF_KERNELS_MEAN_RMS_INTERNALS_HPP
#define _RF_KERNELS_MEAN_RMS_INTERNALS_HPP

#include <simd_helpers/simd_float32.hpp>
#include <simd_helpers/simd_ntuple.hpp>
#include <simd_helpers/udsample.hpp>
#include <simd_helpers/convert.hpp>

#include "mean_rms.hpp"
#include "downsample_internals.hpp"


namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif

template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;


// -------------------------------------------------------------------------------------------------
//
// _wrms_buf: there are different versions of this class corresponding to different memory layouts.
//
// They must define:
//
//    _first_pass(): computes (sum W) and (sum W I)
//
//    _second_pass(): returns sum W (I-mean)^2,  where caller passes mean
//
//    _single_pass(): computes (sum W), (sum W I), and (sum W I^2)
//
//    _iterate(): computes (sum W), (sum W (I-mean)), and (sum W (I-mean)^2),
//                 where caller passes mean.
//
//    get_mask(): for intensity_clipper


template<typename T, int S>
struct _wrms_buf_linear
{
    const T *i_buf;
    const T *w_buf;
    const int bufsize;

    _wrms_buf_linear(const T *i_buf_, const T *w_buf_, int bufsize) : 
	i_buf(i_buf_), 
	w_buf(w_buf_), 
	bufsize(bufsize)
    { }


    inline void _first_pass(simd_t<T,S> &wsum, simd_t<T,S> &wisum) const
    {
	wsum = wisum = simd_t<T,S>::zero();
	
	for (int it = 0; it < bufsize; it += S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + it);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + it);

	    wisum += wval * ival;
	    wsum += wval;
	}
    }

    
    inline simd_t<T,S> _second_pass(simd_t<T,S> mean) const
    {
	simd_t<T,S> wiisum = simd_t<T,S>::zero();
	
	for (int it = 0; it < bufsize; it += S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + it);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + it);
	    
	    ival -= mean;
	    wiisum += wval * ival * ival;
	}

	return wiisum;
    }


    inline void _single_pass(simd_t<T,S> &wsum, simd_t<T,S> &wisum, simd_t<T,S> &wiisum) const
    {
	wsum = wisum = wiisum = simd_t<T,S>::zero();
	
	for (int it = 0; it < bufsize; it += S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + it);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + it);
	    
	    simd_t<T,S> wival = wval * ival;
	    wiisum += wival * ival;
	    wisum += wival;
	    wsum += wval;
	}
    }

    
    inline void _iterate(simd_t<T,S> &wsum, simd_t<T,S> &wisum, simd_t<T,S> &wiisum, simd_t<T,S> mean, simd_t<T,S> thresh) const
    {
	wsum = wisum = wiisum = simd_t<T,S>::zero();
	
	for (int it = 0; it < bufsize; it += S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + it);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + it);
	    
	    // Use mean from previous iteration to (hopefully!) improve numerical stability.
	    ival -= mean;
	    
	    // Note: use of "<" here (rather than "<=") means that we always mask if thresh=0.
	    simd_t<T,S> valid = (ival.abs() < thresh);
	    wval &= valid;
	    
	    simd_t<T,S> wival = wval * ival;
	    wiisum += wival * ival;
	    wisum += wival;
	    wsum += wval;
	}
    }
    

    // For intensity_clipper.
    inline simd_t<T,S> get_mask(simd_t<T,S> mean, simd_t<T,S> thresh, int i) const
    {
	simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + i);
	ival -= mean;

	// Note: use of "<" here (rather than "<=") means that we always mask if thresh=0.
	simd_t<T,S> valid = (ival.abs() < thresh);
	return valid;
    }

};


template<typename T, int S>
struct _wrms_buf_scattered
{
    const T *i_buf;
    const T *w_buf;
    const int nfreq;
    const int stride;

    _wrms_buf_scattered(const T *i_buf_, const T *w_buf_, int nfreq_, int stride_) : 
	i_buf(i_buf_), 
	w_buf(w_buf_), 
	nfreq(nfreq_),
	stride(stride_)
    { }


    inline void _first_pass(simd_t<T,S> &wsum, simd_t<T,S> &wisum) const
    {
	wsum = wisum = simd_t<T,S>::zero();

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + ifreq*stride);

	    wisum += wval * ival;
	    wsum += wval;
	}
    }

    
    inline simd_t<T,S> _second_pass(simd_t<T,S> mean) const
    {
	simd_t<T,S> wiisum = simd_t<T,S>::zero();

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + ifreq*stride);
	    
	    ival -= mean;
	    wiisum += wval * ival * ival;
	}

	return wiisum;
    }


    inline void _single_pass(simd_t<T,S> &wsum, simd_t<T,S> &wisum, simd_t<T,S> &wiisum) const
    {
	wsum = wisum = wiisum = simd_t<T,S>::zero();

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + ifreq*stride);
	    
	    simd_t<T,S> wival = wval * ival;
	    wiisum += wival * ival;
	    wisum += wival;
	    wsum += wval;
	}
    }

    
    inline void _iterate(simd_t<T,S> &wsum, simd_t<T,S> &wisum, simd_t<T,S> &wiisum, simd_t<T,S> mean, simd_t<T,S> thresh) const
    {
	wsum = wisum = wiisum = simd_t<T,S>::zero();

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + ifreq*stride);
	    
	    // Use mean from previous iteration to (hopefully!) improve numerical stability.
	    ival -= mean;
	    
	    // Note: use of "<" here (rather than "<=") means that we always mask if thresh=0.
	    simd_t<T,S> valid = (ival.abs() < thresh);
	    wval &= valid;
	    
	    simd_t<T,S> wival = wval * ival;
	    wiisum += wival * ival;
	    wisum += wival;
	    wsum += wval;
	}
    }
    

    // For intensity_clipper.
    inline simd_t<T,S> get_mask(simd_t<T,S> mean, simd_t<T,S> thresh, int ifreq) const
    {
	simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride);
	ival -= mean;

	// Note: use of "<" here (rather than "<=") means that we always mask if thresh=0.
	simd_t<T,S> valid = (ival.abs() < thresh);
	return valid;
    }
};


template<typename T, int S>
struct _wrms_buf_strided
{
    const T *i_buf;
    const T *w_buf;
    const int nfreq;
    const int nt_chunk;
    const int stride;

    _wrms_buf_strided(const T *i_buf_, const T *w_buf_, int nfreq_, int nt_chunk_, int stride_) :
	i_buf(i_buf_),
	w_buf(w_buf_),
	nfreq(nfreq_),
	nt_chunk(nt_chunk_),
	stride(stride_)
    { }

    
    inline void _first_pass(simd_t<T,S> &wsum, simd_t<T,S> &wisum) const
    {
	wsum = wisum = simd_t<T,S>::zero();

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    for (int it = 0; it < nt_chunk; it += S) {
		simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride + it);
		simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + ifreq*stride + it);

		wisum += wval * ival;
		wsum += wval;
	    }
	}
    }

    
    inline simd_t<T,S> _second_pass(simd_t<T,S> mean) const
    {
	simd_t<T,S> wiisum = simd_t<T,S>::zero();

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    for (int it = 0; it < nt_chunk; it += S) {
		simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride + it);
		simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + ifreq*stride + it);
	    
		ival -= mean;
		wiisum += wval * ival * ival;
	    }
	}

	return wiisum;
    }


    inline void _single_pass(simd_t<T,S> &wsum, simd_t<T,S> &wisum, simd_t<T,S> &wiisum) const
    {
	wsum = wisum = wiisum = simd_t<T,S>::zero();

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    for (int it = 0; it < nt_chunk; it += S) {
		simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride + it);
		simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + ifreq*stride + it);
	    
		simd_t<T,S> wival = wval * ival;
		wiisum += wival * ival;
		wisum += wival;
		wsum += wval;
	    }
	}
    }

    
    inline void _iterate(simd_t<T,S> &wsum, simd_t<T,S> &wisum, simd_t<T,S> &wiisum, simd_t<T,S> mean, simd_t<T,S> thresh) const
    {
	wsum = wisum = wiisum = simd_t<T,S>::zero();

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    for (int it = 0; it < nt_chunk; it += S) {
		simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride + it);
		simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + ifreq*stride + it);
	    
		// Use mean from previous iteration to (hopefully!) improve numerical stability.
		ival -= mean;
		
		// Note: use of "<" here (rather than "<=") means that we always mask if thresh=0.
		simd_t<T,S> valid = (ival.abs() < thresh);
		wval &= valid;
		
		simd_t<T,S> wival = wval * ival;
		wiisum += wival * ival;
		wisum += wival;
		wsum += wval;
	    }
	}
    }
    

    // For intensity_clipper.
    inline simd_t<T,S> get_mask(simd_t<T,S> mean, simd_t<T,S> thresh, int ifreq, int it) const
    {
	simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + ifreq*stride + it);
	ival -= mean;

	// Note: use of "<" here (rather than "<=") means that we always mask if thresh=0.
	simd_t<T,S> valid = (ival.abs() < thresh);
	return valid;
    }
};


// -------------------------------------------------------------------------------------------------
//
// _wrms_first_pass
//
// Defines
//
//   finalize(buf, mean, var) -> call after downsampler
//   run(buf, mean, var) -> to run from scratch


template<bool Hflag, typename T, int S, typename std::enable_if<Hflag,int>::type = 0>
inline void _hsum(simd_t<T,S> &x) { x = x.horizontal_sum(); }

template<bool Hflag, typename T, int S, typename std::enable_if<(!Hflag),int>::type = 0>
inline void _hsum(simd_t<T,S> &x) { }


template<typename T, int S, bool Hflag, bool TwoPass>
struct _wrms_first_pass;


// TwoPass = true
template<typename T, int S, bool Hflag>
struct _wrms_first_pass<T,S,Hflag,true>
{
    simd_t<T,S> wsum;
    simd_t<T,S> wisum;

    _wrms_first_pass()
    {
	wsum = simd_t<T,S>::zero();
	wisum = simd_t<T,S>::zero();
    }

    // Callback for _wi_downsampler.
    inline void ds_put(simd_t<T,S> ival, simd_t<T,S> wval, simd_t<T,S> wival)
    {
	wisum += wival;
	wsum += wval;
    }

    // Called after _wi_downsampler.
    template<typename Tbuf>
    inline void finalize(const Tbuf &buf, simd_t<T,S> &mean, simd_t<T,S> &var)
    {
	const simd_t<T,S> zero = 0.0;
	const simd_t<T,S> one = 1.0;
	
	_hsum<Hflag> (wisum);
	_hsum<Hflag> (wsum);
	
	simd_t<T,S> wsum_reg = blendv(wsum.compare_gt(zero), wsum, one);
	simd_t<T,S> wden = one / wsum_reg;
	
	// Second pass to compute variance.
	mean = wden * wisum;
	simd_t<T,S> wiisum = buf._second_pass(mean);
	
	_hsum<Hflag> (wiisum);
	var = wden * wiisum;

	// Threshold variance at (eps_2 mean)^2.
	constexpr T eps_2 = 1.0e2 * simd_helpers::machine_epsilon<T> ();
	simd_t<T,S> cutoff = simd_t<T,S>(eps_2) * mean;
	simd_t<T,S> valid = (var >= cutoff*cutoff);   // decided to use ">=" here (not ">")
	var &= valid;
    }

    // Run from scratch.
    template<typename Tbuf>
    inline void run(const Tbuf &buf, simd_t<T,S> &mean, simd_t<T,S> &var)
    {
	buf._first_pass(wsum, wisum);
	this->finalize(buf, mean, var);
    }
};


// TwoPass = false
template<typename T, int S, bool Hflag>
struct _wrms_first_pass<T,S,Hflag,false>
{
    simd_t<T,S> wsum;
    simd_t<T,S> wisum;
    simd_t<T,S> wiisum;

    _wrms_first_pass()
    {
	wsum = simd_t<T,S>::zero();
	wisum = simd_t<T,S>::zero();
	wiisum = simd_t<T,S>::zero();
    }
    
    // Callback for _wi_downsampler.
    inline void ds_put(simd_t<T,S> ival, simd_t<T,S> wval, simd_t<T,S> wival)
    {
	wsum += wval;
	wisum += wival;
	wiisum += wival * ival;
    }

    // Called after _wi_downsampler.
    template<typename Tbuf>
    inline void finalize(const Tbuf &buf, simd_t<T,S> &mean, simd_t<T,S> &var)
    {
	const simd_t<T,S> zero(0.0);
	const simd_t<T,S> one(1.0);

	_hsum<Hflag> (wsum);
	_hsum<Hflag> (wisum);
	_hsum<Hflag> (wiisum);
	
	simd_t<T,S> wsum_reg = blendv(wsum.compare_gt(zero), wsum, one);
	simd_t<T,S> wden = one / wsum_reg;
	
	mean = wden * wisum;
	var = wden * wiisum - mean * mean;

	// Threshold variance at (eps_3 mean^2).
	constexpr T eps_3 = 1.0e3 * simd_helpers::machine_epsilon<T> ();
	simd_t<T,S> cutoff = simd_t<T,S>(eps_3) * mean * mean;
	simd_t<T,S> valid = (var >= cutoff);   // decided to use ">=" here (not ">")
	var &= valid;
    }

    // Run from scratch.
    template<typename Tbuf>
    inline void run(const Tbuf &buf, simd_t<T,S> &mean, simd_t<T,S> &var)
    {
	buf._single_pass(wsum, wisum, wiisum);
	this->finalize(buf, mean, var);
    }
};
    

// -------------------------------------------------------------------------------------------------
//
// _wrms_iterate
//
// Note: caller should call with (niter-1)!


template<bool Hflag, typename Tbuf, typename T, int S>
inline void _wrms_iterate(Tbuf &buf, simd_t<T,S> &mean, simd_t<T,S> &var, int niter, simd_t<T,S> sigma)
{
    const simd_t<T,S> zero = 0;
    const simd_t<T,S> one = 1;

    for (int iter = 0; iter < niter; iter++) {
	simd_t<T,S> thresh = var.sqrt() * sigma;
	
	simd_t<T,S> wsum, wisum, wiisum;
	buf._iterate(wsum, wisum, wiisum, mean, thresh);

	_hsum<Hflag> (wsum);
	_hsum<Hflag> (wisum);
	_hsum<Hflag> (wiisum);
	
	// FIXME need epsilons here?
	simd_t<T,S> wsum_reg = blendv(wsum.compare_gt(zero), wsum, one);
	simd_t<T,S> wden = one / wsum_reg;
	simd_t<T,S> dmean = wden * wisum;
	    
	// Don't update mean yet (wait until after thresholding)
	var = wden * wiisum - dmean*dmean;

	// Threshold variance at (eps_2 in_mean)^2.
	constexpr T eps_2 = 1.0e2 * simd_helpers::machine_epsilon<T> ();
	simd_t<T,S> cutoff = simd_t<T,S>(eps_2) * mean;
	simd_t<T,S> valid = (var >= cutoff*cutoff);   // decided to use ">=" here (not ">")
	var &= valid;

	// Threshold variance at (eps_3 dmean^2).
	constexpr T eps_3 = 1.0e3 * simd_helpers::machine_epsilon<T> ();
	cutoff = simd_t<T,S>(eps_3) * dmean * dmean;
	valid = (var >= cutoff);   // decided to use ">=" here (not ">")
	var &= valid;

	// Update mean.
	mean += dmean;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Low-level wrms kernels.


template<typename T, int S, int DfX, int DtX, bool TwoPass, typename std::enable_if<((DfX>1)||(DtX>1)),int>::type = 0>
inline void kernel_wrms_taxis(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int stride)
{
    constexpr int Hflag = true;
    
    const int Df = wp->Df;
    const int nfreq_ds = wp->nfreq_ds;
    const int nt_ds = wp->nt_ds;
    const int niter = wp->niter;
    const simd_t<T,S> sigma = wp->sigma;

    float *tmp_i = wp->tmp_i;
    float *tmp_w = wp->tmp_w;
    float *out_mean = wp->out_mean;
    float *out_rms = wp->out_rms;

    _wrms_buf_linear<T,S> buf(tmp_i, tmp_w, nt_ds);
    _wi_downsampler_1d<T,S,DfX,DtX> ds1(Df, wp->Dt);
	
    for (int ifreq = 0; ifreq < nfreq_ds; ifreq++) {
	_wrms_first_pass<T,S,Hflag,TwoPass> fp;
	
	ds1.downsample_1d(fp, nt_ds, stride,
			  in_i + ifreq * Df * stride,
			  in_w + ifreq * Df * stride,
			  tmp_i, tmp_w);

	simd_t<T,S> mean, var;
	fp.finalize(buf, mean, var);

	_wrms_iterate<Hflag> (buf, mean, var, niter-1, sigma);

	simd_t<T,S> rms = var.sqrt();
	out_mean[ifreq] = mean.template extract<0> ();
	out_rms[ifreq] = rms.template extract<0> ();
    }
}


template<typename T, int S, int DfX, int DtX, bool TwoPass, typename std::enable_if<((DfX>1)||(DtX>1)),int>::type = 0>
inline void kernel_wrms_faxis(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int stride)
{
    constexpr int Hflag = false;
    
    const int Df = wp->Df;
    const int Dt = wp->Dt;
    const int niter = wp->niter;
    const int nt_ds = wp->nt_ds;
    const int nfreq_ds = wp->nfreq_ds;
    const simd_t<T,S> sigma = wp->sigma;

    float *tmp_i = wp->tmp_i;
    float *tmp_w = wp->tmp_w;
    float *out_mean = wp->out_mean;
    float *out_rms = wp->out_rms;
    
    _wrms_buf_linear<T,S> buf(tmp_i, tmp_w, nfreq_ds*S);
    _wi_downsampler_1f<T,S,DfX,DtX> ds1(Df, Dt);

    for (int it = 0; it < nt_ds; it += S) {
	_wrms_first_pass<T,S,Hflag,TwoPass> fp;
	
	ds1.downsample_1f(fp, nfreq_ds, stride,
			  in_i + it*Dt,
			  in_w + it*Dt,
			  tmp_i, tmp_w);

	simd_t<T,S> mean, var;
	fp.finalize(buf, mean, var);

	_wrms_iterate<Hflag> (buf, mean, var, niter-1, sigma);

	simd_helpers::simd_store(out_mean + it, mean);
	simd_helpers::simd_store(out_rms + it, var.sqrt());
    }	
}


template<typename T, int S, int DfX, int DtX, bool TwoPass, typename std::enable_if<((DfX>1)||(DtX>1)),int>::type = 0>
inline void kernel_wrms_naxis(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int stride)
{
    constexpr int Hflag = true;
    
    const int Df = wp->Df;
    const int nfreq_ds = wp->nfreq_ds;
    const int nt_ds = wp->nt_ds;
    const int niter = wp->niter;
    const simd_t<T,S> sigma = wp->sigma;

    float *tmp_i = wp->tmp_i;
    float *tmp_w = wp->tmp_w;
    float *out_mean = wp->out_mean;
    float *out_rms = wp->out_rms;

    _wi_downsampler_1d<T,S,DfX,DtX> ds1(Df, wp->Dt);
    _wrms_buf_linear<T,S> buf(tmp_i, tmp_w, nfreq_ds * nt_ds);
    _wrms_first_pass<T,S,Hflag,TwoPass> fp;
    
    for (int ifreq = 0; ifreq < nfreq_ds; ifreq++) {	
	ds1.downsample_1d(fp, nt_ds, stride,
			  in_i + ifreq * Df * stride,
			  in_w + ifreq * Df * stride,
			  tmp_i + ifreq * nt_ds,
			  tmp_w + ifreq * nt_ds);
    }

    simd_t<T,S> mean, var;
    fp.finalize(buf, mean, var);

    _wrms_iterate<Hflag> (buf, mean, var, niter-1, sigma);

    simd_t<T,S> rms = var.sqrt();
    out_mean[0] = mean.template extract<0> ();
    out_rms[0] = rms.template extract<0> ();
}


// -------------------------------------------------------------------------------------------------


template<typename T, int S, int DfX, int DtX, bool TwoPass, typename std::enable_if<((DfX==1)&&(DtX==1)),int>::type = 0>
inline void kernel_wrms_taxis(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int stride)
{
    constexpr int Hflag = true;
    
    const int nfreq = wp->nfreq;
    const int nt = wp->nt_chunk;
    const int niter = wp->niter;
    const simd_t<T,S> sigma = wp->sigma;

    float *out_mean = wp->out_mean;
    float *out_rms = wp->out_rms;
	
    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	_wrms_buf_linear<T,S> buf(in_i + ifreq*stride, in_w + ifreq*stride, nt);
	_wrms_first_pass<T,S,Hflag,TwoPass> fp;

	simd_t<T,S> mean, var;
	fp.run(buf, mean, var);

	_wrms_iterate<Hflag> (buf, mean, var, niter-1, sigma);

	simd_t<T,S> rms = var.sqrt();
	out_mean[ifreq] = mean.template extract<0> ();
	out_rms[ifreq] = rms.template extract<0> ();
    }
}

template<typename T, int S, int DfX, int DtX, bool TwoPass, typename std::enable_if<((DfX==1)&&(DtX==1)),int>::type = 0>
inline void kernel_wrms_faxis(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int stride)
{
    constexpr int Hflag = false;
    
    const int niter = wp->niter;
    const int nt = wp->nt_chunk;
    const int nfreq = wp->nfreq;
    const simd_t<T,S> sigma = wp->sigma;

    float *out_mean = wp->out_mean;
    float *out_rms = wp->out_rms;
    
    for (int it = 0; it < nt; it += S) {
	_wrms_buf_scattered<T,S> buf(in_i + it, in_w + it, nfreq, stride);
	_wrms_first_pass<T,S,Hflag,TwoPass> fp;

	simd_t<T,S> mean, var;
	fp.run(buf, mean, var);

	_wrms_iterate<Hflag> (buf, mean, var, niter-1, sigma);

	simd_helpers::simd_store(out_mean + it, mean);
	simd_helpers::simd_store(out_rms + it, var.sqrt());
    }	
}


template<typename T, int S, int DfX, int DtX, bool TwoPass, typename std::enable_if<((DfX==1)&&(DtX==1)),int>::type = 0>
inline void kernel_wrms_naxis(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int stride)
{
    constexpr int Hflag = true;
    
    const int nfreq = wp->nfreq;
    const int nt = wp->nt_chunk;
    const int niter = wp->niter;
    const simd_t<T,S> sigma = wp->sigma;

    float *out_mean = wp->out_mean;
    float *out_rms = wp->out_rms;

    _wrms_buf_strided<T,S> buf(in_i, in_w, nfreq, nt, stride);
    _wrms_first_pass<T,S,Hflag,TwoPass> fp;

    simd_t<T,S> mean, var;
    fp.run(buf, mean, var);

    _wrms_iterate<Hflag> (buf, mean, var, niter-1, sigma);

    simd_t<T,S> rms = var.sqrt();
    out_mean[0] = mean.template extract<0> ();
    out_rms[0] = rms.template extract<0> ();
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_MEAN_RMS_INTERNALS_HPP
