#ifndef _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP
#define _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP

#include <cmath>
#include <simd_helpers/simd_float32.hpp>

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


// AXIS_TIME
template<typename T, int S, int DfX, int DtX>
inline void kernel_std_dev_clipper(std_dev_clipper *sd, const T *in_i, T *in_w, int stride)
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
	const T *in_i2 = in_i + ifreq_ds * Df * stride;
	T *in_w2 = in_w + ifreq_ds * Df * stride;

	_wrms_1d_outbuf<T,S> out(tmp_i, tmp_w, nt_ds);

	ds1.downsample_1d(out, nt_ds, in_i2, in_w2, stride);
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


}  // namespace rf_kernels

#endif  // _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP
