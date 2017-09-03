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


// Currently assume AXIS_TIME
// Currently assume two_pass=true
template<typename T, int S>
struct _wrms_1d_outbuf {
    const simd_t<T,S> zero = 0;
    const simd_t<T,S> one = 1;

    T *i_out;
    T *w_out;
    
    const int nds_t;
    
    simd_t<T,S> wisum = 0;
    simd_t<T,S> wsum = 0;
    
    simd_t<T,S> mean;
    simd_t<T,S> var;


    _wrms_1d_outbuf(T *i_out_, T *w_out_, int nds_t_) : 
	i_out(i_out_), 
	w_out(w_out_), 
	nds_t(nds_t_)
    { }


    // Callback for _wi_downsampler.
    inline void put(simd_t<T,S> wival, simd_t<T,S> wval, int it)
    {
	wisum += wival;
	wsum += wval;
	
	// FIXME revisit after smask cleanup.
	wival /= blendv(wval.compare_gt(zero), wval, one);
	wival.storeu(i_out + it);
	wval.storeu(w_out + it);	
    }


    // Called after wi_downsampler, to finalize variance.
    inline void finalize(int niter, simd_t<T,S> sigma)
    {
	wisum = wisum.horizontal_sum();
	wsum = wsum.horizontal_sum();
	
	simd_t<T,S> wsum_reg = blendv(wsum.compare_gt(zero), wsum, one);
	simd_t<T,S> wden = one / wsum_reg;
	
	mean = wden * wisum;
	simd_t<T,S> wiisum = zero;

	// Second pass to compute rms.
	for (int it = 0; it < nds_t; it += S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_out + it);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_out + it);
	    
	    ival -= mean;
	    wiisum += wval * ival * ival;
	}

	// FIXME need epsilons here?
	var = wden * wiisum.horizontal_sum();

	// Note (niter-1) iterations here (not niter iterations)
	for (int iter = 1; iter < niter; iter++) {
	    simd_t<T,S> thresh = var.sqrt() * sigma;

	    wiisum = zero;
	    wisum = zero;
	    wsum = zero;
	
	    for (int it = 0; it < nds_t; it += S) {
		simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_out + it);
		simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_out + it);
		
		// Use mean from previous iteration to (hopefully!) improve numerical stability.
		ival -= mean;
		
		simd_t<T,S> valid = (ival.abs() <= thresh);
		wval &= valid;
		
		simd_t<T,S> wival = wval * ival;
		wiisum += wival * ival;
		wisum += wival;
		wsum += wval;
	    }
	
	    wiisum = wiisum.horizontal_sum();
	    wisum = wisum.horizontal_sum();
	    wsum = wsum.horizontal_sum();

	    // FIXME need epsilons here?
	    wsum_reg = blendv(wsum.compare_gt(zero), wsum, one);
	    wden = one / wsum_reg;
	    simd_t<T,S> dmean = wden * wisum;
	    
	    mean += dmean;
	    var = wden * wiisum - dmean*dmean;
	}
    }


    // For intensity clipper!
    inline simd_t<T,S> get_mask(simd_t<T,S> thresh, int it)
    {
	simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_out + it);
	ival -= mean;

	simd_t<T,S> valid = (ival.abs() <= thresh);
	return valid;
    }
};


// -------------------------------------------------------------------------------------------------


// Note: Still assuming (axis, two_pass) = (AXIS_TIME, false).
template<typename T, int S, int DfX, int DtX>
inline void kernel_wrms(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int stride)
{
    const int Df = wp->Df;
    const int nfreq_ds = wp->nfreq_ds;
    const int nt_ds = wp->nt_ds;
    const int niter = wp->niter;
    const simd_t<T,S> sigma = wp->sigma;
    const _wi_downsampler_1d<T,S,DfX,DtX> ds1(Df, wp->Dt);

    float *tmp_i = wp->tmp_i;
    float *tmp_w = wp->tmp_w;
    float *out_mean = wp->out_mean;
    float *out_rms = wp->out_rms;

    for (int ifreq = 0; ifreq < nfreq_ds; ifreq++) {
	T *out_i2 = tmp_i + ifreq * nt_ds;
	T *out_w2 = tmp_w + ifreq * nt_ds;
	const T *in_i2 = in_i + ifreq * Df * stride;
	const T *in_w2 = in_w + ifreq * Df * stride;

	_wrms_1d_outbuf<T,S> out(out_i2, out_w2, nt_ds);
	ds1.downsample_1d(out, nt_ds, in_i2, in_w2, stride);

	out.finalize(niter, sigma);

	simd_t<T,S> rms = out.var.sqrt();
	out_mean[ifreq] = out.mean.template extract<0> ();
	out_rms[ifreq] = rms.template extract<0> ();
    }
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_MEAN_RMS_INTERNALS_HPP
