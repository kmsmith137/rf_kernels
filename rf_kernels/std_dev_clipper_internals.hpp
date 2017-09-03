#ifndef _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP
#define _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP

#include <cmath>
#include <simd_helpers/simd_float32.hpp>

#include "downsample_internals.hpp"
#include "mean_rms_internals.hpp"


namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// Defined in std_dev_clippers.cpp
extern void clip_1d(int n, float *tmp_sd, int *tmp_valid, double sigma);


template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;
template<typename T, int S> using smask_t = simd_helpers::smask_t<T,S>;
template<typename T, int S, int D> using simd_ntuple = simd_helpers::simd_ntuple<T,S,D>;
template<typename T, int S, int D> using smask_ntuple = simd_helpers::smask_ntuple<T,S,D>;


// -------------------------------------------------------------------------------------------------
//
// _kernel_mask1<T,S,R,N> (T *weights, smask_t<T,S> mask, int stride)
//
// Masks a strided array of shape (R,N*S), by repeating a single simd_t<T,S>.


template<typename T, int S, int R, int N, typename std::enable_if<(R==0 || N==0),int>::type = 0>
inline void _kernel_mask1(T *weights, smask_t<T,S> mask, int stride)
{
    return;
}


template<typename T, int S, int R, int N, typename std::enable_if<(R>0 && N>0),int>::type = 0>
inline void _kernel_mask1(T *weights, smask_t<T,S> mask, int stride)
{
    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (weights);
    wval = wval.apply_mask(mask);
    wval.storeu(weights);

    _kernel_mask1<T,S,R-1,1> (weights+stride, mask, stride);
    _kernel_mask1<T,S,R,N-1> (weights+S, mask, stride);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_mask2<T,S,R,D,N> (T *weights, smask_ntuple<T,S,D> &mask, int stride)
//
// Masks a strided array of shape (R,D*N*S), by repeating the mask over the (r,n) axes.


template<typename T, int S, int R, int D, int N, typename std::enable_if<(D==0),int>::type = 0>
inline void _kernel_mask2(T *weights, const smask_ntuple<T,S,D> &mask, int stride)
{
    return;
}


template<typename T, int S, int R, int D, int N, typename std::enable_if<(D>0),int>::type = 0>
inline void _kernel_mask2(T *weights, const smask_ntuple<T,S,D> &mask, int stride)
{
    _kernel_mask2<T,S,R,D-1,N> (weights, mask.v, stride);
    _kernel_mask1<T,S,R,N> (weights + (D-1)*N*S, mask.x, stride);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_mask<T,S,R,D> (T *weights, smask_t<T,S> mask, int stride)
//
// Upsamples mask by a factor (R,D) in (freq,time) directions, and applies it to the shape-(R,D*S)
// strided array based at 'weights'.


template<typename T, int S, int R, int D, typename std::enable_if<(D==1),int>::type = 0>
inline void _kernel_mask(T *weights, smask_t<T,S> mask, int stride)
{
    _kernel_mask1<T,S,R,1> (weights, mask, stride);
}


template<typename T, int S, int R, int D, typename std::enable_if<(D>1 && D<=S),int>::type = 0>
inline void _kernel_mask(T *weights, smask_t<T,S> mask, int stride)
{
    smask_ntuple<T,S,D> masku;
    simd_helpers::upsample(masku, mask);

    _kernel_mask2<T,S,R,D,1> (weights, masku, stride);
}


template<typename T, int S, int R, int D, typename std::enable_if<(D>S),int>::type = 0>
inline void _kernel_mask(T *weights, smask_t<T,S> mask, int stride)
{
    smask_ntuple<T,S,S> masku;
    simd_helpers::upsample(masku, mask);

    _kernel_mask2<T,S,R,S,D/S> (weights, masku, stride);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_mask_columns<T,S,Dt> (T *weights, smask_t<T,1> *mask, int nfreq, int nt, int stride)
//
// This kernel is applied to a 2D strided array with shape (nfreq, nt).
// The mask is a 1D array of length nt/Dt, which is upsampled by a factor Dt and applied.
// Caller must check that nt % (Dt*S) == 0.


template<typename T, int S, int Dt>
inline void _kernel_mask_columns(T *weights, const smask_t<T,1> *mask, int nfreq, int nt, int stride)
{
    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	T *wrow = weights + ifreq * stride;
	const smask_t<T,1> *mrow = mask;

	for (int it = 0; it < nt; it += Dt*S) {
	    smask_t<T,S> m;
	    m.loadu(mrow);
	    _kernel_mask<T,S,1,Dt> (wrow + it, m, stride);
	    mrow += S;
	}
    }
}


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
    _kernel_std_dev_f<T,S,Df,Dt,TwoPass> (intensity, weights, nfreq, nt, stride, out_sd, out_valid, ds_int, ds_wt);

    clip_1d(nt/Dt, out_sd, out_valid, sigma);

    _kernel_mask_columns<T,S,Dt> (weights, out_valid, nfreq, nt, stride);
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_STD_DEV_CLIPPER_INTERNALS_HPP
