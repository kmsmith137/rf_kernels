#ifndef _RF_KERNELS_DOWNSAMPLE_INTERNALS_HPP
#define _RF_KERNELS_DOWNSAMPLE_INTERNALS_HPP

#include <simd_helpers/simd_float32.hpp>
#include <simd_helpers/simd_ntuple.hpp>
#include <simd_helpers/udsample.hpp>
#include <simd_helpers/downsample.hpp>

#include "downsample.hpp"


namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif

template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;
template<typename T, int S, int D> using simd_ntuple = simd_helpers::simd_ntuple<T,S,D>;
template<typename T, int S, int D> using simd_downsampler = simd_helpers::simd_downsampler<T,S,D>;


// -------------------------------------------------------------------------------------------------
//
// _kernel_downsample1<T,S,R,N> (simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
//
// Reads a strided array of shape (R,N*S), and sums the result over outer index r
// and middle index N, returning a simd_t<T,S>.


// The "1a" variant accumulates its result
template<typename T, int S, int R, int N, typename std::enable_if<(R==0 || N==0),int>::type = 0>
inline void _kernel_downsample1a(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    return;
}


template<typename T, int S, int R, int N, typename std::enable_if<(R > 0 && N > 0),int>::type = 0>
inline void _kernel_downsample1a(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (intensity);
    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (weights);

    ds_wi += wval * ival;
    ds_w += wval;

    _kernel_downsample1a<T,S,R-1,1> (ds_wi, ds_w, intensity+stride, weights+stride, stride);
    _kernel_downsample1a<T,S,R,N-1> (ds_wi, ds_w, intensity+S, weights+S, stride);
}


template<typename T, int S, int R, int N, typename std::enable_if<(R > 0 && N > 0),int>::type = 0>
inline void _kernel_downsample1(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (intensity);
    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (weights);
    
    ds_wi = wval * ival;
    ds_w = wval;
    
    _kernel_downsample1a<T,S,R-1,1> (ds_wi, ds_w, intensity+stride, weights+stride, stride);
    _kernel_downsample1a<T,S,R,N-1> (ds_wi, ds_w, intensity+S, weights+S, stride);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_downsample2<T,S,R,D,N> (simd_ntuple<T,S,D> &ds_wi, simd_ntuple<T,S,D> &ds_w, const T *intensity, const T *weights, int stride)
//
// Reads a strided array of shape (R,D*N*S), and sums the result over outer index r
// and middle index n, returning a simd_ntuple<T,S,D>.


template<typename T, int S, int R, int D, int N, typename std::enable_if<(D==0),int>::type = 0>
inline void _kernel_downsample2(simd_ntuple<T,S,D> &ds_wi, simd_ntuple<T,S,D> &ds_w, const T *intensity, const T *weights, int stride)
{
    return;
}


template<typename T, int S, int R, int D, int N, typename std::enable_if<(D>0),int>::type = 0>
inline void _kernel_downsample2(simd_ntuple<T,S,D> &ds_wi, simd_ntuple<T,S,D> &ds_w, const T *intensity, const T *weights, int stride)
{
    _kernel_downsample2<T,S,R,D-1,N> (ds_wi.v, ds_w.v, intensity, weights, stride);
    _kernel_downsample1<T,S,R,N> (ds_wi.x, ds_w.x, intensity + (D-1)*N*S, weights + (D-1)*N*S, stride);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_downsample<T,S,R,D> (simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
//
// Reads a strided array of shape (R,D*S), and sums the result over outer index r and inner index d, 
// returning a simd_t<T,S>.

 
template<typename T, int S, int R, int D, typename std::enable_if<(D==1),int>::type = 0>
inline void _kernel_downsample(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    _kernel_downsample1<T,S,R,1> (ds_wi, ds_w, intensity, weights, stride);
}


template<typename T, int S, int R, int D, typename std::enable_if<(D>1 && D<=S),int>::type = 0>
inline void _kernel_downsample(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    simd_ntuple<T,S,D> dsn_wi, dsn_w;
    _kernel_downsample2<T,S,R,D,1> (dsn_wi, dsn_w, intensity, weights, stride);

    ds_wi = simd_helpers::downsample(dsn_wi);   // defined in simd_helpers/udsample.hpp
    ds_w = simd_helpers::downsample(dsn_w);
}


template<typename T, int S, int R, int D, typename std::enable_if<(D>S),int>::type = 0>
inline void _kernel_downsample(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    simd_ntuple<T,S,S> dsn_wi, dsn_w;
    _kernel_downsample2<T,S,R,S,D/S> (dsn_wi, dsn_w, intensity, weights, stride);

    ds_wi = simd_helpers::downsample(dsn_wi);
    ds_w = simd_helpers::downsample(dsn_w);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_downsample_2d<T,S,Df,Dt> (out_intensity, out_weights, out_stride, in_intensity, in_nfreq, in_nt, in_stride)
//
// Caller must check that nfreq is divisible by Df, and nt is divisible by (Dt*S).
//
// This is the kernel which gets called in the externally visible function wi_downsample().


template<typename T, int S, int Df, int Dt>
inline void _kernel_downsample_2d(T *out_intensity, T *out_weights, int out_stride, const T *in_intensity, const T *in_weights, int in_nfreq, int in_nt, int in_stride)
{
    const simd_t<T,S> zero = simd_t<T,S>::zero();
    const simd_t<T,S> one = simd_t<T,S> (1.0);

    int out_nfreq = in_nfreq / Df;
    int out_nt = in_nt / Dt;

    for (int ifreq = 0; ifreq < out_nfreq; ifreq++) {
	T *out_irow = out_intensity + ifreq * out_stride;
	T *out_wrow = out_weights + ifreq * out_stride;

	const T *in_irow = in_intensity + (ifreq*Df) * in_stride;
	const T *in_wrow = in_weights + (ifreq*Df) * in_stride;

	for (int it = 0; it < out_nt; it += S) {
	    simd_t<T,S> ds_wival, ds_wval;
	    _kernel_downsample<T,S,Df,Dt> (ds_wival, ds_wval, in_irow + it*Dt, in_wrow + it*Dt, in_stride);

	    simd_t<T,S> ds_ival = ds_wival / blendv(ds_wval.compare_gt(zero), ds_wval, one);
	    ds_ival.storeu(out_irow + it);
	    ds_wval.storeu(out_wrow + it);
	}
    }
}


// -------------------------------------------------------------------------------------------------
//
// _wi_downsample_0a<Df> (wi_acc, w_acc, intensity, weights, stride)
// _wi_downsample_0a<Df> (wi_acc, w_acc, intensity, weights, stride, N)
//
// These sum over a shape-(Df,S) and shape-(Df,N) array respectively.
// In the latter case, caller must check that N is divisible by the simd size S!
// No downsampling kernels are used -- the sum is purely "vertical".


template<int Df, typename T, int S, typename std::enable_if<(Df==0),int>::type = 0>
inline void _wi_downsample_0a(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, const T *in_w, int stride)
{ }

template<int Df, typename T, int S, typename std::enable_if<(Df>0),int>::type = 0>
inline void _wi_downsample_0a(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, const T *in_w, int stride)
{
    _wi_downsample_0a<Df-1> (wi_acc, w_acc, in_i, in_w, stride);
    
    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (in_i + (Df-1)*stride);
    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (in_w + (Df-1)*stride);

    wi_acc += wval * ival;
    w_acc += wval;
}

template<int Df, typename T, int S>
inline void _wi_downsample_0a(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, const T *in_w, int stride, int N)
{
    for (int it = 0; it < N; it += S)
	_wi_downsample_0a<Df> (wi_acc, w_acc, in_i + it, in_w + it, stride);
}
			      
// ----------------------------------------------------------------------------------------------------
//
// _wi_downsampler_0d<T, S, Df0, Dt> ds0(Dt_);
//
// simd_t<T,S> wi_out, w_out;
// ds0.get(wi_out, w_out, i_in, w_in, stride);
//
// We use the notation Df0 here to emphasize that in a higher-dimensional context, Df may not be equal to Df0!


template<typename T, int S, int Df0, int Dt, bool Dt_Large = (Dt > S)>
struct _wi_downsampler_0d;


// Case 1: "small" Dt
template<typename T, int S, int Df0, int Dt>
struct _wi_downsampler_0d<T, S, Df0, Dt, false> 
{
    _wi_downsampler_0d(int Dt_)
    {
	if (__builtin_expect(Dt != Dt_, 0))
	    throw std::runtime_error("rf_kernels internal error: \"small\" Dt mismatch in _wi_downsampler_0d");
    }
    
    constexpr int get_Dt() const { return Dt; }

    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,Dt> &wi_ds, simd_downsampler<T,S,Dt> &w_ds, const T *in_i, const T *in_w, int stride) const
    { }

    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,Dt> &wi_ds, simd_downsampler<T,S,Dt> &w_ds, const T *in_i, const T *in_w, int stride) const
    {
	_get_partial<P-1> (wi_ds, w_ds, in_i, in_w, stride);
	
	simd_t<T,S> wi_acc = simd_t<T,S>::zero();
	simd_t<T,S> w_acc = simd_t<T,S>::zero();
	_wi_downsample_0a<Df0> (wi_acc, w_acc, in_i + (P-1)*S, in_w + (P-1)*S, stride);
	
	wi_ds.template put<P-1> (wi_acc);
	w_ds.template put<P-1> (w_acc);
    }

    
    inline void get(simd_t<T,S> &wi_out, simd_t<T,S> &w_out, const T *i_in, const T *w_in, int stride) const
    {
	simd_downsampler<T,S,Dt> wi_ds, w_ds;
	_get_partial<Dt> (wi_ds, w_ds, i_in, w_in, stride);

	wi_out += wi_ds.get();
	w_out += w_ds.get();
    }
};


// Case 2: "Large" Dt
template<typename T, int S, int Df0, int DtX>
struct _wi_downsampler_0d<T, S, Df0, DtX, true>
{
    const int Dt;

    _wi_downsampler_0d(int Dt_) : 
	Dt(Dt_) 
    { 
	if (__builtin_expect((Dt_ <= S) || (Dt_ % S), 0))
	    throw std::runtime_error("rf_kernels internal error: bad \"large\" Dt in _wi_downsampler_0d");
    }

    inline int get_Dt() const { return Dt; }

    
    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,S> &wi_ds, simd_downsampler<T,S,S> &w_ds, const T *in_i, const T *in_w, int stride) const
    { }
    
    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,S> &wi_ds, simd_downsampler<T,S,S> &w_ds, const T *in_i, const T *in_w, int stride) const
    {
	_get_partial<P-1> (wi_ds, w_ds, in_i, in_w, stride);
	
	simd_t<T,S> wi_acc = simd_t<T,S>::zero();
	simd_t<T,S> w_acc = simd_t<T,S>::zero();
	_wi_downsample_0a<Df0> (wi_acc, w_acc, in_i + (P-1)*Dt, in_w + (P-1)*Dt, stride, Dt);
	
	wi_ds.template put<P-1> (wi_acc);
	w_ds.template put<P-1> (w_acc);    
    }

    
    inline void get(simd_t<T,S> &wi_out, simd_t<T,S> &w_out, const T *i_in, const T *w_in, int stride) const
    {
	simd_downsampler<T,S,S> wi_ds, w_ds;
	_get_partial<S> (wi_ds, w_ds, i_in, w_in, stride);
	
	wi_out += wi_ds.get();
	w_out += w_ds.get();
    }
};


// -------------------------------------------------------------------------------------------------
//
// _wi_downsampler_1d<T, S, Df, Dt> ds1(Df_, Dt_);
//
// _wi_downsampler_1d_outbuf<T, S> out;
//
// downsample_1d(out, nt_out, in_i, in_w, stride);


template<typename T, int S, int Df, int Dt, bool Df_Large = (Df > S)>
struct _wi_downsampler_1d;


// Case 1: "small" Df.
template<typename T, int S, int Df, int DtX>
struct _wi_downsampler_1d<T, S, Df, DtX, false> 
{
    const _wi_downsampler_0d<T, S, Df, DtX> ds0;

    _wi_downsampler_1d(int Df_, int Dt_) :
	ds0(Dt_)
    {
	if (__builtin_expect(Df != Df_, 0))
	    throw std::runtime_error("rf_kernels internal error: \"small\" Df mismatch in _wi_downsampler_1d");
    }

    template<typename Tout>
    inline void downsample_1d(Tout &out, int nt_out, const T *i_in, const T *w_in, int istride) const
    {
	const int Dt = ds0.get_Dt();
	
	simd_t<T,S> wival;
	simd_t<T,S> wval;
    
	for (int it = 0; it < nt_out; it += S) {
	    wival = simd_t<T,S>::zero();
	    wval = simd_t<T,S>::zero();

	    ds0.get(wival, wval, i_in + it*Dt, w_in + it*Dt, istride);
	    out.put(wival, wval, it);
	}
    }
};


// Case 2: "Large" Df.
template<typename T, int S, int DfX, int DtX>
struct _wi_downsampler_1d<T, S, DfX, DtX, true> 
{
    const _wi_downsampler_0d<T, S, S, DtX> ds0;
    const int Df;
    
    _wi_downsampler_1d(int Df_, int Dt_) :
	ds0(Dt_),
	Df(Df_)
    {
	if (__builtin_expect((Df_ <= S) || (Df_ % S), 0))
	    throw std::runtime_error("rf_kernels internal error: bad \"large\" Df in _wi_downsampler_1d");
    }

    template<typename Tout>
    inline void downsample_1d(Tout &out, int nt_out, const T *i_in, const T *w_in, int istride) const
    {
	const int Dt = ds0.get_Dt();

	T *i_out = out.i_out;
	T *w_out = out.w_out;
	
	simd_t<T,S> wival;
	simd_t<T,S> wval;

	// First pass
	for (int it = 0; it < nt_out; it += S) {
	    wival = simd_t<T,S>::zero();
	    wval = simd_t<T,S>::zero();

	    ds0.get(wival, wval, i_in + it*Dt, w_in + it*Dt, istride);
	    wival.storeu(i_out + it);
	    wval.storeu(w_out + it);
	}

	i_in += S*istride;
	w_in += S*istride;
	
	// Middle passes
	for (int i = S; i < (Df-S); i += S) {
	    for (int it = 0; it < nt_out; it += S) {
		wival.loadu(i_out + it);
		wval.loadu(w_out + it);
		
		ds0.get(wival, wval, i_in + it*Dt, w_in + it*Dt, istride);
		wival.storeu(i_out + it);
		wval.storeu(w_out + it);
	    }
	    
	    i_in += S*istride;
	    w_in += S*istride;
	}

	// Last pass
	for (int it = 0; it < nt_out; it += S) {
	    wival.loadu(i_out + it);
	    wval.loadu(w_out + it);
	    
	    ds0.get(wival, wval, i_in + it*Dt, w_in + it*Dt, istride);
	    out.put(wival, wval, it);
	}
    }
};


// 1D "outbuf" class
template<typename T, int S>
struct _wi_downsampler_1d_outbuf {
    T *i_out;
    T *w_out;

    _wi_downsampler_1d_outbuf(T *i_out_, T *w_out_) : i_out(i_out_), w_out(w_out_) { }
    
    const simd_t<T,S> zero = 0;
    const simd_t<T,S> one = 1;

    inline void put(simd_t<T,S> wival, simd_t<T,S> wval, int it)
    {
	// FIXME revisit after smask cleanup.
	wival /= blendv(wval.compare_gt(zero), wval, one);
	wival.storeu(i_out + it);
	wval.storeu(w_out + it);	
    }
};


// -------------------------------------------------------------------------------------------------


template<typename T, int S, int DfX, int DtX, typename std::enable_if<((DfX > 1) || (DtX > 1)),int>::type = 0>
inline void kernel_wi_downsample(const wi_downsampler *dp, int nfreq_out, int nt_out, T *out_i, T *out_w, int ostride, const T *in_i, const T *in_w, int istride)
{
    const int Df = dp->Df;
    const _wi_downsampler_1d<T,S,DfX,DtX> ds1(Df, dp->Dt);
    
    for (int ifreq = 0; ifreq < nfreq_out; ifreq++) {
	_wi_downsampler_1d_outbuf<T,S> out(out_i + ifreq*ostride, out_w + ifreq*ostride);
	ds1.downsample_1d(out, nt_out, in_i + ifreq*Df*istride, in_w + ifreq*Df*istride, istride);
    }
}


// Special case (Df,Dt)=(1,1).
template<typename T, int S, int DfX, int DtX, typename std::enable_if<((DfX == 1) && (DtX == 1)),int>::type = 0>
inline void kernel_wi_downsample(const wi_downsampler *dp, int nfreq_out, int nt_out, T *out_i, T *out_w, int ostride, const T *in_i, const T *in_w, int istride)
{
    for (int ifreq = 0; ifreq < nfreq_out; ifreq++)
	memcpy(out_i + ifreq*ostride, in_i + ifreq*istride, nt_out * sizeof(T));
    
    for (int ifreq = 0; ifreq < nfreq_out; ifreq++)
	memcpy(out_w + ifreq*ostride, in_w + ifreq*istride, nt_out * sizeof(T));
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_DOWNSAMPLE_INTERNALS_HPP
