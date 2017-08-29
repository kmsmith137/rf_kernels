#ifndef _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP
#define _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP

#include <simd_helpers/simd_float32.hpp>

#include "downsample_internals.hpp"
#include "clipper_internals.hpp"

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// Defined in std_dev_clippers.cpp
extern void clip_1d(int n, float *tmp_sd, int *tmp_valid, double sigma);


// -------------------------------------------------------------------------------------------------
//
// _kernel_std_dev_t()
//
// This is the "bottom line" routine called by std_dev_clipper(AXIS_TIME).
//
// Caller must check (nfreq % Df) == 0 and (nt % (Dt*S)) == 0.
// There is no requirement that (nfreq % (Df*S)) == 0.


template<typename T, int S, int Df, int Dt, bool TwoPass>
inline void _kernel_std_dev_t(const T *intensity, const T *weights, int nfreq, int nt, int stride, T *out_sd, smask_t<T,1> *out_valid, T *ds_int, T *ds_wt)
{
    for (int ifreq = 0; ifreq < nfreq; ifreq += Df) {
	const T *irow = intensity + ifreq * stride;
	const T *wrow = weights + ifreq * stride;

	simd_t<T,S> mean, var;
	_kernel_mean_variance_1d_t<T,S,Df,Dt,false,false,TwoPass> (mean, var, irow, wrow, nt, stride, ds_int, ds_wt);

	// scalar instructions should be fine here
	T sd = var.template extract<0> ();
	*out_sd++ = sd;
	*out_valid++ = (sd > 0.0) ? smask_t<T,1>(-1) : 0;
    }
}


template<typename T, int S, int Df, int Dt, bool TwoPass>
inline void _kernel_std_dev_clip_time_axis(const T *intensity, T *weights, int nfreq, int nt, int stride, double sigma, T *out_sd, smask_t<T,1> *out_valid, T *ds_int, T *ds_wt)
{
    _kernel_std_dev_t<T,S,Df,Dt,TwoPass> (intensity, weights, nfreq, nt, stride, out_sd, out_valid, ds_int, ds_wt);

    clip_1d(nfreq/Df, out_sd, out_valid, sigma);

    for (int i = 0; i < nfreq/Df; i++) {
	if (out_valid[i])
	    continue;

	for (int ifreq = i*Df; ifreq < (i+1)*Df; ifreq++)
	    memset(weights + ifreq*stride, 0, nt * sizeof(T));
    }
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_std_dev_f()
//
// This is the "bottom line" routine called by std_dev_clipper(AXIS_FREQ).
//
// Caller must check (nfreq % Df) == 0 and (nt % (Dt*S)) == 0.
// There is no requirement that (nfreq % (Df*S)) == 0.


template<typename T, int S, int Df, int Dt, bool TwoPass>
inline void _kernel_std_dev_f(const T *intensity, const T *weights, int nfreq, int nt, int stride, T *out_sd, smask_t<T,1> *out_valid, T *ds_int, T *ds_wt)
{
    const simd_t<T,S> zero = simd_t<T,S>::zero();

    for (int it = 0; it < nt; it += Dt*S) {
	const T *icol = intensity + it;
	const T *wcol = weights + it;

	simd_t<T,S> mean, var;
	_kernel_mean_variance_1d_f<T,S,Df,Dt,false,false,TwoPass> (mean, var, icol, wcol, nfreq, stride, ds_int, ds_wt);
	
	smask_t<T,S> valid = var.compare_gt(zero);

	var.storeu(out_sd);
	valid.storeu(out_valid);

	out_sd += S;
	out_valid += S;
    }
}


template<typename T, int S, int Df, int Dt, bool TwoPass>
inline void _kernel_std_dev_clip_freq_axis(const T *intensity, T *weights, int nfreq, int nt, int stride, double sigma, T *out_sd, smask_t<T,1> *out_valid, T *ds_int, T *ds_wt)
{
    _kernel_std_dev_f<T,S,Df,Dt,TwoPass> (intensity, weights, nfreq, nt, stride, out_sd, out_valid, ds_int);

    clip_1d(nt/Dt, out_sd, outvalid, sigma);

    _kernel_mask_columns<T,S,Dt> (weights, out_valid, nfreq, nt, stride);
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP
