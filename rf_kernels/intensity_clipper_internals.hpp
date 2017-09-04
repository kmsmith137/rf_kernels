#ifndef _RF_KERNELS_INTENSITY_CLIPPER_INTERNALS_HPP
#define _RF_KERNELS_INTENSITY_CLIPPER_INTERNALS_HPP

#include "upsample_internals.hpp"
#include "downsample_internals.hpp"
#include "mean_rms_internals.hpp"
#include "intensity_clipper.hpp"

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif

template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;


template<typename T, int S, int DfX, int DtX>
inline void kernel_intensity_clipper_taxis(const intensity_clipper *ic, const T *in_i, T *in_w, int stride)
{
    // For upsampler.  Note std::min() can't be used in a constexpr!
    constexpr int Dfu = (DfX < 8) ? DfX : 8;

    const int Df = ic->Df;
    const int Dt = ic->Dt;
    const int nfreq_ds = ic->nfreq_ds;
    const int nt_ds = ic->nt_ds;
    const int niter = ic->niter;
    const simd_t<T,S> sigma(ic->sigma);
    const simd_t<T,S> iter_sigma(ic->iter_sigma);

    float *tmp_i = ic->tmp_i;
    float *tmp_w = ic->tmp_w;
    
    _wi_downsampler_1d<T, S, DfX, DtX> ds1(Df, Dt);
    _weight_upsampler_0d<T, S, Dfu, DtX> us0(Dt);

    for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++) {
	_wrms_1d_outbuf<T,S,AXIS_TIME> out(tmp_i, tmp_w, nt_ds);
	ds1.downsample_1d(out, nt_ds, in_i + ifreq_ds*Df*stride, in_w + ifreq_ds*Df*stride, stride);

	// Note iter_sigma here (not sigma)	
	out.finalize(niter, iter_sigma);
	
	// Note sigma here (not iter_sigma)
	simd_t<T,S> thresh = sigma * out.var.sqrt();

	// Mask rows (note that we mask "Dfu" rows at a time, where Dfu = min(Df,8).
	for (int ifreq_u = ifreq_ds*Df; ifreq_u < (ifreq_ds+1)*Df; ifreq_u += Dfu) {
	    T *wrow = in_w + ifreq_u * stride;

	    for (int it = 0; it < nt_ds; it += S) {
		simd_t<T,S> mask = out.get_mask(thresh, it);
		us0.put_mask(wrow + it*Dt, stride, mask);
	    }
	}
    }
}


template<typename T, int S, int DfX, int DtX>
inline void kernel_intensity_clipper_faxis(const intensity_clipper *ic, const T *in_i, T *in_w, int stride)
{
    const int Df = ic->Df;
    const int Dt = ic->Dt;
    const int nfreq_ds = ic->nfreq_ds;
    const int nt_ds = ic->nt_ds;
    const int niter = ic->niter;
    const simd_t<T,S> sigma(ic->sigma);
    const simd_t<T,S> iter_sigma(ic->iter_sigma);

    float *tmp_i = ic->tmp_i;
    float *tmp_w = ic->tmp_w;
    
    _wi_downsampler_1f<T, S, DtX> ds1(Df, Dt);
    _weight_upsampler_0d<T, S, 1, DtX> us0(Dt);  // FIXME to be improved soon!

    for (int it_ds = 0; it_ds < nt_ds; it_ds += S) {
	_wrms_1d_outbuf<T,S,AXIS_FREQ> out(tmp_i, tmp_w, nfreq_ds);
	ds1.downsample_1f(out, nfreq_ds, in_i + it_ds*Dt, in_w + it_ds*Dt, stride);

	// Note iter_sigma here (not sigma)
	out.finalize(niter, iter_sigma);
	
	// Note sigma here (not iter_sigma)
	simd_t<T,S> thresh = sigma * out.var.sqrt();

	for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++) {
	    simd_t<T,S> mask = out.get_mask(thresh, ifreq_ds);

	    for (int ifreq_us = ifreq_ds*Df; ifreq_us < (ifreq_ds+1)*Df; ifreq_us++)
		us0.put_mask(in_w + ifreq_us*stride + it_ds*Dt, stride, mask);
	}
    }	
}


}  // namespace rf_kernels

#endif // _RF_KERNELS_INTENSITY_CLIPPER_INTERNALS_HPP
