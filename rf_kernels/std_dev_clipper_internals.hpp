#ifndef _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP
#define _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP

#include <cmath>
#include <simd_helpers/simd_float32.hpp>

#include "upsample_internals.hpp"
#include "downsample_internals.hpp"
#include "mean_rms_internals.hpp"
#include "std_dev_clipper.hpp"


namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif

template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;


// Defined in std_dev_clippers.cpp
extern void clip_1d(int n, float *tmp_sd, int *tmp_valid, double sigma);


template<typename T, int S, int DfX, int DtX>
inline void kernel_std_dev_clipper_taxis(std_dev_clipper *sd, const T *in_i, T *in_w, int stride)
{
    const int Df = sd->Df;
    const int Dt = sd->Dt;
    const int nt_chunk = sd->nt_chunk;
    const int nfreq_ds = sd->nfreq_ds;
    const int nt_ds = sd->nt_ds;

    float *tmp_i = sd->tmp_i;
    float *tmp_w = sd->tmp_w;
    float *tmp_v = sd->tmp_v;
    
    _wi_downsampler_1d<T, S, DfX, DtX> ds1(Df, Dt);

    for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++) {
	_wrms_1d_outbuf<T,S,AXIS_TIME> out(tmp_i, tmp_w, nt_ds);
	ds1.downsample_1d(out, nt_ds, in_i + ifreq_ds*Df*stride, in_w + ifreq_ds*Df*stride, stride);

	out.finalize(1, out.zero);
	tmp_v[ifreq_ds] = out.var.template extract<0> ();
    }

    sd->_clip_1d();

    for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++) {
	if (tmp_v[ifreq_ds] > 0.0)
	    continue;

	for (int ifreq = ifreq_ds*Df; ifreq < (ifreq_ds+1)*Df; ifreq++)
	    memset(in_w + ifreq*stride, 0, nt_chunk * sizeof(T));
    }
}


template<typename T, int S, int DfX, int DtX>
inline void kernel_std_dev_clipper_faxis(std_dev_clipper *sd, const T *in_i, T *in_w, int stride)
{
    const int Df = sd->Df;
    const int Dt = sd->Dt;
    const int nfreq_us = sd->nfreq;
    const int nfreq_ds = sd->nfreq_ds;
    const int nt_ds = sd->nt_ds;

    float *tmp_i = sd->tmp_i;
    float *tmp_w = sd->tmp_w;
    float *tmp_v = sd->tmp_v;

    _wi_downsampler_1f<T, S, DtX> ds1(Df, Dt);

    for (int it_ds = 0; it_ds < nt_ds; it_ds += S) {
	_wrms_1d_outbuf<T,S,AXIS_FREQ> out(tmp_i, tmp_w, nfreq_ds);
	ds1.downsample_1f(out, nfreq_ds, in_i + it_ds*Dt, in_w + it_ds*Dt, stride);

	// niter=1, sigma=zero (placeholder)
	out.finalize(1, out.zero);
	simd_helpers::simd_store(tmp_v + it_ds, out.var);
    }

    sd->_clip_1d();

    // Now we upsample and apply the mask.
    // We make frequency the outer loop, and mask 8 rows at a time.
    // This scheme should minimize worst-case running time (but suboptimal in the
    // case where not much data is masked).
    //
    // Note: we assume that nfreq (not nfreq_ds) is a mutiple of 8.
    // This is checked in the std_dev_clipper constructor.

    constexpr int R = 8;
    _weight_upsampler_0d<T, S, R, DtX> us0(Dt);

    for (int ifreq_us = 0; ifreq_us < nfreq_us; ifreq_us += R) {
	for (int it_ds = 0; it_ds < nt_ds; it_ds += S) {
	    simd_t<T,S> v = simd_helpers::simd_load<T,S> (tmp_v + it_ds);
	    simd_t<T,S> mask = (v > simd_t<T,S>::zero());
	    us0.put_mask(in_w + ifreq_us*stride + it_ds*Dt, stride, mask);
	}
    }
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP
