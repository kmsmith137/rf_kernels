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


template<bool Hflag, typename T, int S, typename std::enable_if<Hflag,int>::type = 0>
inline void _hsum(simd_t<T,S> &x) { x = x.horizontal_sum(); }

template<bool Hflag, typename T, int S, typename std::enable_if<(!Hflag),int>::type = 0>
inline void _hsum(simd_t<T,S> &x) { }


// Currently assume two_pass=true
template<typename T, int S, bool Hflag>
struct _wrms_buf_linear {
    T *i_buf;
    T *w_buf;
    const int bufsize;
    
    simd_t<T,S> wisum;
    simd_t<T,S> wsum;
    
    simd_t<T,S> mean;
    simd_t<T,S> var;


    _wrms_buf_linear(T *i_buf_, T *w_buf_, int bufsize) : 
	i_buf(i_buf_), 
	w_buf(w_buf_), 
	bufsize(bufsize)
    { }


    // Callback for _wi_downsampler.
    inline void ds_init()
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


    // Called after _wi_downsampler, to horizontally sum and finalize variance.
    inline void finalize()
    {
	const simd_t<T,S> zero = 0;
	const simd_t<T,S> one = 1;
	
	_hsum<Hflag> (wisum);
	_hsum<Hflag> (wsum);
	
	simd_t<T,S> wsum_reg = blendv(wsum.compare_gt(zero), wsum, one);
	simd_t<T,S> wden = one / wsum_reg;
	
	mean = wden * wisum;
	simd_t<T,S> wiisum = zero;

	// Second pass to compute rms.
	for (int it = 0; it < bufsize; it += S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + it);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_buf + it);
	    
	    ival -= mean;
	    wiisum += wval * ival * ival;
	}

	_hsum<Hflag> (wiisum);
	
	// FIXME need epsilons here?
	var = wden * wiisum;
    }


    // Note: should be called with (niter-1)
    inline void iterate(int niter, simd_t<T,S> sigma)
    {
	const simd_t<T,S> zero = 0;
	const simd_t<T,S> one = 1;

	for (int iter = 0; iter < niter; iter++) {
	    simd_t<T,S> thresh = var.sqrt() * sigma;
	    simd_t<T,S> wiisum = zero;

	    wisum = zero;
	    wsum = zero;
	
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

	    _hsum<Hflag> (wsum);
	    _hsum<Hflag> (wisum);
	    _hsum<Hflag> (wiisum);

	    // FIXME need epsilons here?
	    simd_t<T,S> wsum_reg = blendv(wsum.compare_gt(zero), wsum, one);
	    simd_t<T,S> wden = one / wsum_reg;
	    simd_t<T,S> dmean = wden * wisum;
	    
	    mean += dmean;
	    var = wden * wiisum - dmean*dmean;
	}
    }


    // For intensity clipper!
    inline simd_t<T,S> get_mask(simd_t<T,S> thresh, int i)
    {
	simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_buf + i);
	ival -= mean;

	// Note: use of "<" here (rather than "<=") means that we always mask if thresh=0.
	simd_t<T,S> valid = (ival.abs() < thresh);
	return valid;
    }
};


// -------------------------------------------------------------------------------------------------
//
// Low-level wrms kernels.


template<typename T, int S, int DfX, int DtX>
inline void kernel_wrms_taxis(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int stride)
{
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
    _wrms_buf_linear<T,S,true> out(tmp_i, tmp_w, nt_ds);
	
    for (int ifreq = 0; ifreq < nfreq_ds; ifreq++) {	
	ds1.downsample_1d(out, nt_ds, stride,
			  in_i + ifreq * Df * stride,
			  in_w + ifreq * Df * stride,
			  tmp_i, tmp_w);

	out.finalize();
	out.iterate(niter-1, sigma);

	simd_t<T,S> rms = out.var.sqrt();
	out_mean[ifreq] = out.mean.template extract<0> ();
	out_rms[ifreq] = rms.template extract<0> ();
    }
}


template<typename T, int S, int DtX>
inline void kernel_wrms_faxis(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int stride)
{
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
    
    _wi_downsampler_1f<T,S,DtX> ds1(Df, Dt);
    _wrms_buf_linear<T,S,false> out(tmp_i, tmp_w, nfreq_ds*S);

    for (int it = 0; it < nt_ds; it += S) {
	ds1.downsample_1f(out, nfreq_ds, stride,
			  in_i + it*Dt,
			  in_w + it*Dt,
			  tmp_i, tmp_w);

	out.finalize();
	out.iterate(niter-1, sigma);

	simd_helpers::simd_store(out_mean + it, out.mean);
	simd_helpers::simd_store(out_rms + it, out.var.sqrt());
    }	
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_MEAN_RMS_INTERNALS_HPP
