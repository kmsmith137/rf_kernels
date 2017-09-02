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
template<typename T, int S> using smask_t = simd_helpers::smask_t<T,S>;


// -------------------------------------------------------------------------------------------------
//
// "Visit" kernels: these loop over a 2D or 1D array with on-the-fly downsampling, 
// applying a "visitor" class V.
//
// Note that either 
//    - V::accumulate_i() will be called with an (intensity,weights) pair (I,W), or
//    - V::accumulate_wi() will be called with (W*I,W)
//
// The former happens if the array is not downsampled, i.e. (Df,Dt)=(1,1).
//
// Some possibilities for V are defined later in this file:
//    _mean_variance_visitor
//    _mean_visitor
//    _variance_visitor
//    _mean_variance_iterator
//
// Note: here and elsewhere in kernels/*.hpp, we define three kernel versions
// (with suffixes _2d, _1d_t, _1d_f).  This seemed unnecessary and I wanted to clean
// up by only using the 2d version (with a boolean flag to enable the horizontal_sum()).
// Unfortunately testing showed that this slowed down the kernels, even if compiled
// with -funroll-loops!  (See 'abandoned_kernel_refactoring' github branch.)


template<int Df, int Dt, typename V, typename std::enable_if<((Df>1) || (Dt>1)),int>::type = 0>
inline void _kernel_visit_2d(V &v, const typename V::T *intensity, const typename V::T *weights, int nfreq, int nt, int stride)
{
    using T = typename V::T;
    constexpr int S = V::S;

    for (int ifreq = 0; ifreq < nfreq; ifreq += Df) {
	const T *irow = intensity + ifreq*stride;
	const T *wrow = weights + ifreq*stride;

	for (int it = 0; it < nt; it += Dt*S) {
	    simd_t<T,S> wival, wval;
	    _kernel_downsample<T,S,Df,Dt> (wival, wval, irow + it, wrow + it, stride);

	    v.accumulate_wi(wival, wval);
	}
    }

    v.horizontal_sum();
}


template<int Df, int Dt, typename V, typename std::enable_if<((Df==1) && (Dt==1)),int>::type = 0>
inline void _kernel_visit_2d(V &v, const typename V::T *intensity, const typename V::T *weights, int nfreq, int nt, int stride)
{
    using T = typename V::T;
    constexpr int S = V::S;

    for (int ifreq = 0; ifreq < nfreq; ifreq += Df) {
	const T *irow = intensity + ifreq*stride;
	const T *wrow = weights + ifreq*stride;

	for (int it = 0; it < nt; it += S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S>(irow + it);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S>(wrow + it);

	    v.accumulate_i(ival, wval);
	}
    }

    v.horizontal_sum();
}


template<int Df, int Dt, typename V, typename std::enable_if<((Df>1) || (Dt>1)),int>::type = 0>
inline void _kernel_visit_1d_f(V &v, const typename V::T *intensity, const typename V::T *weights, int nfreq, int stride)
{
    using T = typename V::T;
    constexpr int S = V::S;

    for (int ifreq = 0; ifreq < nfreq; ifreq += Df) {
	simd_t<T,S> wival, wval;
	_kernel_downsample<T,S,Df,Dt> (wival, wval, intensity + ifreq*stride, weights + ifreq*stride, stride);
	v.accumulate_wi(wival, wval);
    }
}


template<int Df, int Dt, typename V, typename std::enable_if<((Df==1) && (Dt==1)),int>::type = 0>
inline void _kernel_visit_1d_f(V &v, const typename V::T *intensity, const typename V::T *weights, int nfreq, int stride)
{
    using T = typename V::T;
    constexpr int S = V::S;

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	simd_t<T,S> ival = simd_helpers::simd_load<T,S>(intensity + ifreq*stride);
	simd_t<T,S> wval = simd_helpers::simd_load<T,S>(weights + ifreq*stride);
	v.accumulate_i(ival, wval);
    }
}


// -------------------------------------------------------------------------------------------------
//
// _mean_variance_visitor
//
// In this and other visitors, the (Iflag,Wflag) compile-time args control whether the downsampled
// intensity and weights arrays (computed on-the-fly) are also written out to auxiliary arrays.


template<typename T_, int S_, bool Iflag, bool Wflag>
struct _mean_variance_visitor {
    using T = T_;
    static constexpr int S = S_;

    const simd_t<T,S> zero;
    const simd_t<T,S> one;

    simd_t<T,S> acc0;
    simd_t<T,S> acc1;
    simd_t<T,S> acc2;

    T *ds_intensity;
    T *ds_weights;

    _mean_variance_visitor(T *ds_intensity_, T *ds_weights_) :
	zero(simd_t<T,S>::zero()), one(simd_t<T,S>(1.0))
    {
	acc0 = simd_t<T,S>::zero();
	acc1 = simd_t<T,S>::zero();
	acc2 = simd_t<T,S>::zero();

	ds_intensity = ds_intensity_;
	ds_weights = ds_weights_;
    }

    inline void accumulate_i(simd_t<T,S> ival, simd_t<T,S> wval) 
    { 
	simd_t<T,S> wival = wval * ival;

	acc0 += wval; 
	acc1 += wival;
	acc2 += wival * ival;

	// Branches will be optimized out at compile-time
	if (Iflag) {
	    ival.storeu(ds_intensity);
	    ds_intensity += S;
	}
	
	if (Wflag) {
	    wval.storeu(ds_weights);
	    ds_weights += S;
	}
    }

    inline void accumulate_wi(simd_t<T,S> wival, simd_t<T,S> wval)
    {
	simd_t<T,S> ival = wival / blendv(wval.compare_gt(zero), wval, one);

	acc0 += wval;
	acc1 += wival;
	acc2 += wival * ival;

	// Branches will be optimized out at compile-time
	if (Iflag) {
	    ival.storeu(ds_intensity);
	    ds_intensity += S;
	}
	
	if (Wflag) {
	    wval.storeu(ds_weights);
	    ds_weights += S;
	}
    }

    inline void horizontal_sum()
    {
	acc0 = acc0.horizontal_sum();
	acc1 = acc1.horizontal_sum();
	acc2 = acc2.horizontal_sum();
    }

    inline void get_mean_variance(simd_t<T,S> &mean, simd_t<T,S> &var) const
    {
	static constexpr T eps_3 = 1.0e3 * simd_helpers::machine_epsilon<T> ();

	smask_t<T,S> valid = acc0.compare_gt(zero);
	simd_t<T,S> t0 = blendv(valid, acc0, one);
	mean = acc1 / t0;

	simd_t<T,S> mean2 = mean * mean;
	var = acc2/t0 - mean2;

	simd_t<T,S> thresh = simd_t<T,S>(eps_3) * mean2;

	valid = valid.bitwise_and(var.compare_gt(thresh));
	var = var.apply_mask(valid);
    }

    inline void get_mean_rms(simd_t<T,S> &mean, simd_t<T,S> &rms) const
    {
	simd_t<T,S> variance;
	get_mean_variance(mean, variance);
	rms = variance.sqrt();
    }
};


// -------------------------------------------------------------------------------------------------
//
// _mean_visitor


template<typename T_, int S_, bool Iflag, bool Wflag>
struct _mean_visitor {
    using T = T_;
    static constexpr int S = S_;

    const simd_t<T,S> zero;
    const simd_t<T,S> one;

    simd_t<T,S> acc0;
    simd_t<T,S> acc1;

    T *ds_intensity;
    T *ds_weights;

    _mean_visitor(T *ds_intensity_, T *ds_weights_) :
	zero(simd_t<T,S>::zero()), one(simd_t<T,S>(1.0))
    {
	acc0 = simd_t<T,S>::zero();
	acc1 = simd_t<T,S>::zero();

	ds_intensity = ds_intensity_;
	ds_weights = ds_weights_;
    }

    inline void accumulate_i(simd_t<T,S> ival, simd_t<T,S> wval) 
    { 
	acc0 += wval; 
	acc1 += wval * ival;

	if (Iflag) {
	    ival.storeu(ds_intensity);
	    ds_intensity += S;
	}
	
	if (Wflag) {
	    wval.storeu(ds_weights);
	    ds_weights += S;
	}
    }

    inline void accumulate_wi(simd_t<T,S> wival, simd_t<T,S> wval)
    {
	acc0 += wval;
	acc1 += wival;

	if (Iflag) {
	    simd_t<T,S> ival = wival / blendv(wval.compare_gt(zero), wval, one);
	    ival.storeu(ds_intensity);
	    ds_intensity += S;
	}
	
	if (Wflag) {
	    wval.storeu(ds_weights);
	    ds_weights += S;
	}
    }

    inline void horizontal_sum()
    {
	acc0 = acc0.horizontal_sum();
	acc1 = acc1.horizontal_sum();
    }

    inline simd_t<T,S> get_mean() const
    {
	smask_t<T,S> valid = acc0.compare_gt(zero);
	return acc1 / blendv(valid, acc0, one);
    }
};


// -------------------------------------------------------------------------------------------------
//
// _variance_visitor (assumes mean is known beforehand)


template<typename T_, int S_>
struct _variance_visitor {
    using T = T_;
    static constexpr int S = S_;

    const simd_t<T,S> zero;
    const simd_t<T,S> one;

    simd_t<T,S> acc0;
    simd_t<T,S> acc2;
    simd_t<T,S> in_mean;

    _variance_visitor(simd_t<T,S> in_mean_) :
	zero(simd_t<T,S>::zero()), one(simd_t<T,S>(1.0))
    {
	acc0 = simd_t<T,S>::zero();
	acc2 = simd_t<T,S>::zero();
	in_mean = in_mean_;
    }

    inline void accumulate_i(simd_t<T,S> ival, simd_t<T,S> wval) 
    { 
	ival -= in_mean;
	acc0 += wval; 
	acc2 += wval * ival * ival;
    }

    inline void horizontal_sum()
    {
	acc0 = acc0.horizontal_sum();
	acc2 = acc2.horizontal_sum();
    }

    inline simd_t<T,S> get_variance() const
    {
	static constexpr T eps_2 = 1.0e2 * simd_helpers::machine_epsilon<T> ();

	smask_t<T,S> valid = acc0.compare_gt(zero);
	simd_t<T,S> t0 = blendv(valid, acc0, one);
	simd_t<T,S> var = acc2/t0;

	simd_t<T,S> thresh = simd_t<T,S>(eps_2) * in_mean;
	thresh = thresh * thresh;

	valid = valid.bitwise_and(var.compare_gt(thresh));
	return var.apply_mask(valid);
    }
};


// -------------------------------------------------------------------------------------------------
//
// _mean_variance_iterator: this visitor is used when the weighted mean/rms is computed iteratively 
// with clipping.
//
// The visitor class is constructed using the mean from the previous iteration, and the clip threshold.
// After _kernel_visit() completes, an updated mean/variance is available.


template<typename T_, int S_>
struct _mean_variance_iterator {
    using T = T_;
    static constexpr int S = S_;

    const simd_t<T,S> zero;
    const simd_t<T,S> one;

    simd_t<T,S> in_mean;
    simd_t<T,S> in_thresh;

    simd_t<T,S> acc0;
    simd_t<T,S> acc1;
    simd_t<T,S> acc2;

    _mean_variance_iterator(simd_t<T,S> in_mean_, simd_t<T,S> in_thresh_) :
	zero(simd_t<T,S>::zero()), one(simd_t<T,S>(1.0))
    {
	in_mean = in_mean_;
	in_thresh = in_thresh_;

	acc0 = simd_t<T,S>::zero();
	acc1 = simd_t<T,S>::zero();
	acc2 = simd_t<T,S>::zero();
    }

    inline void accumulate_i(simd_t<T,S> ival, simd_t<T,S> wval)
    {
	ival -= in_mean;
	smask_t<T,S> valid = ival.abs().compare_lt(in_thresh);
	
	wval = wval.apply_mask(valid);
	simd_t<T,S> wival = wval * ival;

	acc0 += wval;
	acc1 += wival;
	acc2 += wival * ival;
    }

    inline void horizontal_sum()
    {
	acc0 = acc0.horizontal_sum();
	acc1 = acc1.horizontal_sum();
	acc2 = acc2.horizontal_sum();
    }

    inline void get_mean_rms(simd_t<T,S> &out_mean, simd_t<T,S> &out_rms)
    {
	static constexpr T eps_2 = 1.0e2 * simd_helpers::machine_epsilon<T> ();
	static constexpr T eps_3 = 1.0e3 * simd_helpers::machine_epsilon<T> ();

	smask_t<T,S> valid = acc0.compare_gt(zero);

	simd_t<T,S> t0 = blendv(valid, acc0, one);
	simd_t<T,S> dmean = acc1/t0;

	simd_t<T,S> dmean2 = dmean * dmean;
	simd_t<T,S> var = acc2/t0 - dmean2;

	simd_t<T,S> thresh1 = simd_t<T,S>(eps_2) * in_mean;
	simd_t<T,S> thresh2 = simd_t<T,S>(eps_3) * dmean2;
	thresh2 = thresh2.max(thresh1 * thresh1);

	valid = valid.bitwise_and(var.compare_gt(thresh2));
	var = var.apply_mask(valid);

	out_mean = in_mean + dmean;
	out_rms = var.sqrt();
    }
};


// -------------------------------------------------------------------------------------------------
//
// _kernel_mean_variance(): computes the mean and variance of an array (noniteratively) 
//  using an appropriate visitor.


// _kernel_mean_variance_2d() case 1: one-pass version
template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass, typename std::enable_if<(!TwoPass),int>::type = 0>
inline void _kernel_mean_variance_2d(simd_t<T,S> &mean, simd_t<T,S> &var, const T *intensity, const T *weights, int nfreq, int nt, int stride, T *ds_intensity, T *ds_weights)
{
    _mean_variance_visitor<T,S,Iflag,Wflag> v(ds_intensity, ds_weights);
    _kernel_visit_2d<Df,Dt> (v, intensity, weights, nfreq, nt, stride);
    v.get_mean_variance(mean, var);
}

// _kernel_mean_variance_2d() case 2: two-pass version, downsampled
template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass, typename std::enable_if<(TwoPass && ((Df>1) || (Dt>1))),int>::type = 0>
inline void _kernel_mean_variance_2d(simd_t<T,S> &mean, simd_t<T,S> &var, const T *intensity, const T *weights, int nfreq, int nt, int stride, T *ds_intensity, T *ds_weights)
{
    _mean_visitor<T,S,true,true> v(ds_intensity, ds_weights);
    _kernel_visit_2d<Df,Dt> (v, intensity, weights, nfreq, nt, stride);
    mean = v.get_mean();

    _variance_visitor<T,S> vv(mean);
    _kernel_visit_2d<1,1> (vv, ds_intensity, ds_weights, nfreq/Df, nt/Dt, nt/Dt);
    var = vv.get_variance();
}

// _kernel_mean_variance_2d() case 3: two-pass version, non-downsampled
template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass, typename std::enable_if<(TwoPass && (Df==1) && (Dt==1)),int>::type = 0>
inline void _kernel_mean_variance_2d(simd_t<T,S> &mean, simd_t<T,S> &var, const T *intensity, const T *weights, int nfreq, int nt, int stride, T *ds_intensity, T *ds_weights)
{
    _mean_visitor<T,S,Iflag,Wflag> v(ds_intensity, ds_weights);
    _kernel_visit_2d<1,1> (v, intensity, weights, nfreq, nt, stride);
    mean = v.get_mean();

    _variance_visitor<T,S> vv(mean);
    _kernel_visit_2d<1,1> (vv, intensity, weights, nfreq, nt, stride);
    var = vv.get_variance();
}


// _kernel_mean_variance_1d_f() case 1: one-pass version
template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass, typename std::enable_if<(!TwoPass),int>::type = 0>
inline void _kernel_mean_variance_1d_f(simd_t<T,S> &mean, simd_t<T,S> &var, const T *intensity, const T *weights, int nfreq, int stride, T *ds_intensity, T *ds_weights)
{
    _mean_variance_visitor<T,S,Iflag,Wflag> v(ds_intensity, ds_weights);
    _kernel_visit_1d_f<Df,Dt> (v, intensity, weights, nfreq, stride);
    v.get_mean_variance(mean, var);
}

// _kernel_mean_variance_1d_f() case 2: two-pass version, downsampled
template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass, typename std::enable_if<(TwoPass && ((Df>1) || (Dt>1))),int>::type = 0>
inline void _kernel_mean_variance_1d_f(simd_t<T,S> &mean, simd_t<T,S> &var, const T *intensity, const T *weights, int nfreq, int stride, T *ds_intensity, T *ds_weights)
{
    _mean_visitor<T,S,true,true> v(ds_intensity, ds_weights);
    _kernel_visit_1d_f<Df,Dt> (v, intensity, weights, nfreq, stride);
    mean = v.get_mean();

    _variance_visitor<T,S> vv(mean);
    _kernel_visit_1d_f<1,1> (vv, ds_intensity, ds_weights, nfreq/Df, S);
    var = vv.get_variance();
}

// _kernel_mean_variance_1d_f() case 3: two-pass version, non-downsampled
template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass, typename std::enable_if<(TwoPass && (Df==1) && (Dt==1)),int>::type = 0>
inline void _kernel_mean_variance_1d_f(simd_t<T,S> &mean, simd_t<T,S> &var, const T *intensity, const T *weights, int nfreq, int stride, T *ds_intensity, T *ds_weights)
{
    _mean_visitor<T,S,Iflag,Wflag> v(ds_intensity, ds_weights);
    _kernel_visit_1d_f<1,1> (v, intensity, weights, nfreq, stride);
    mean = v.get_mean();

    _variance_visitor<T,S> vv(mean);
    _kernel_visit_1d_f<1,1> (vv, intensity, weights, nfreq, stride);    
    var = vv.get_variance();
}


// Placeholder for future expansion
template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass>
inline void _kernel_mean_variance_1d_t(simd_t<T,S> &mean, simd_t<T,S> &var, const T *intensity, const T *weights, int nt, int stride, T *ds_intensity, T *ds_weights)
{
    _kernel_mean_variance_2d<T,S,Df,Dt,Iflag,Wflag,TwoPass> (mean, var, intensity, weights, Df, nt, stride, ds_intensity, ds_weights);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_noniterative_wrms()
//
// Computes the weighted mean and rms of a 2D or 1D strided array,
// with downsampling factors (Df,Dt) in the (frequency,time) axes.
//
// If the weighted mean and rms cannot be computed (e.g. because all weights are zero), then
// rms=0 and mean is arbitrary.  (This behavior is inherited from 'struct mean_rms_accumulator'.)
//
// As the downsampled intensity and weights arrays are computed, they are written to
// 'ds_intensity' and 'ds_weights'.  These are unstrided arrays, i.e. the row stride
// is (nt/Dt).
//
// The Iflag, Wflag template arguments will omit writing the ds_intensity, ds_weights
// arrays if set to 'false'.  In this case, passing a NULL pointer is OK.


template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass>
inline void _kernel_noniterative_wrms_2d(simd_t<T,S> &mean, simd_t<T,S> &rms, const T *intensity, const T *weights, int nfreq, int nt, int stride, T *ds_intensity, T *ds_weights)
{
    _kernel_mean_variance_2d<T,S,Df,Dt,Iflag,Wflag,TwoPass> (mean, rms, intensity, weights, nfreq, nt, stride, ds_intensity, ds_weights);
    rms = rms.sqrt();
}

template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass>
inline void _kernel_noniterative_wrms_1d_f(simd_t<T,S> &mean, simd_t<T,S> &rms, const T *intensity, const T *weights, int nfreq, int stride, T *ds_intensity, T *ds_weights)
{
    _kernel_mean_variance_1d_f<T,S,Df,Dt,Iflag,Wflag,TwoPass> (mean, rms, intensity, weights, nfreq, stride, ds_intensity, ds_weights);
    rms = rms.sqrt();
}

template<typename T, int S, int Df, int Dt, bool Iflag, bool Wflag, bool TwoPass>
inline void _kernel_noniterative_wrms_1d_t(simd_t<T,S> &mean, simd_t<T,S> &rms, const T *intensity, const T *weights, int nt, int stride, T *ds_intensity, T *ds_weights)
{
    _kernel_mean_variance_1d_t<T,S,Df,Dt,Iflag,Wflag,TwoPass> (mean, rms, intensity, weights, nt, stride, ds_intensity, ds_weights);
    rms = rms.sqrt();
}

// -------------------------------------------------------------------------------------------------
//
// _kernel_wrms_iterate(): this is called after _kernel_noniterative_wrms().
//
// Note that there are no downsampling factors (Df,Dt) here.  In the downsampled case, we
// first call _kernel_noniterative_wrms() with Iflag=Wflag=true, so that downsampled arrays
// get written.  These arrays are then used as inputs to _kernel_wrms_iterate(), so from
// the perspective of _kernel_wrms_iterate(), no downsampling needs to be done.
//
// Note: the 'niter' argument to these kernels will be one less than the 'niter' argument
// to intensity_clipper().  This is because the initial call to _kernel_noniterative_wrms()
// counts as one iteration.


template<typename T, int S>
inline void _kernel_wrms_iterate_2d(simd_t<T,S> &mean, simd_t<T,S> &rms, const T *intensity, const T *weights, int nfreq, int nt, int stride, int niter, double iter_sigma)
{
    for (int iter = 1; iter < niter; iter++) {
	simd_t<T,S> thresh = simd_t<T,S>(iter_sigma) * rms;
	_mean_variance_iterator<T,S> v(mean, thresh);
	_kernel_visit_2d<1,1> (v, intensity, weights, nfreq, nt, stride);
	v.get_mean_rms(mean, rms);
    }
}

// Placeholder for future expansion
template<typename T, int S>
inline void _kernel_wrms_iterate_1d_t(simd_t<T,S> &mean, simd_t<T,S> &rms, const T *intensity, const T *weights, int nt, int niter, double iter_sigma)
{
    _kernel_wrms_iterate_2d<T,S> (mean, rms, intensity, weights, 1, nt, 0, niter, iter_sigma);
}

template<typename T, int S>
inline void _kernel_wrms_iterate_1d_f(simd_t<T,S> &mean, simd_t<T,S> &rms, const T *intensity, const T *weights, int nfreq, int stride, int niter, double iter_sigma)
{
    for (int iter = 1; iter < niter; iter++) {
	simd_t<T,S> thresh = simd_t<T,S>(iter_sigma) * rms;
	_mean_variance_iterator<T,S> v(mean, thresh);
	_kernel_visit_1d_f<1,1> (v, intensity, weights, nfreq, stride);	
	v.get_mean_rms(mean, rms);
    }
}


//=================================================================================================
//=================================================================================================
//=================================================================================================
//=================================================================================================


// -------------------------------------------------------------------------------------------------
//
// Usage: kernel(out_mean, nfreq, nt, in_i, in_w, istride, Df, Dt, niter, sigma, tmp_i, tmp_w)


template<typename T, int S>
struct _wrms_1d_outbuf {
    T *i_out;
    T *w_out;
    
    simd_t<T,S> wisum = 0;
    simd_t<T,S> wsum = 0;

    _wrms_1d_outbuf(T *i_out_, T *w_out_) : i_out(i_out_), w_out(w_out_) { }

    const simd_t<T,S> zero = 0;
    const simd_t<T,S> one = 1;

    inline void put(simd_t<T,S> wival, simd_t<T,S> wval, int it)
    {
	wisum += wival;
	wsum += wval;
	
	// FIXME revisit after smask cleanup.
	wival /= blendv(wval.compare_gt(zero), wval, one);
	wival.storeu(i_out + it);
	wval.storeu(w_out + it);	
    }
};


template<typename T, int S, int Df, int Dt>
inline void kernel_wrms_Dfsm_Dtsm(const weighted_mean_rms *wp, const T *in_i, const T *in_w, int istride)
{
    int nfreq_ds = wp->nfreq_ds;
    int nt_ds = wp->nt_ds;
    float *tmp_i = wp->tmp_i;
    float *tmp_w = wp->tmp_w;
    float *out_mean = wp->out_mean;

    _wi_downsampler_0d_Dtsm<T,S,Df,Dt> ds0;
    _wi_downsampler_1d_Dfsm<decltype(ds0)> ds1(ds0);

    for (int ifreq = 0; ifreq < nfreq_ds; ifreq++) {
	T *out_i2 = tmp_i + ifreq * nt_ds;
	T *out_w2 = tmp_w + ifreq * nt_ds;
	const T *in_i2 = in_i + ifreq * Df * istride;
	const T *in_w2 = in_w + ifreq * Df * istride;

	_wrms_1d_outbuf<T,S> out(out_i2, out_w2);
	
	ds1.downsample_1d(out, nt_ds, in_i2, in_w2, istride);

	float wisum = out.wisum.sum();
	float wsum = out.wsum.sum();
	out_mean[ifreq] = (wsum > 0.0) ? (wisum/wsum) : 0.0;
    }
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_MEAN_RMS_INTERNALS_HPP
