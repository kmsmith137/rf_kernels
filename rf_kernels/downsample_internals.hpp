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
template<typename T, int S, int N> using simd_ntuple = simd_helpers::simd_ntuple<T,S,N>;
template<typename T, int S, int D> using simd_downsampler = simd_helpers::simd_downsampler<T,S,D>;


// -------------------------------------------------------------------------------------------------
//
// _sum_strided<Df> (wi_acc, w_acc, intensity, weights, stride)
// _sum_strided<Df> (wi_acc, w_acc, intensity, weights, stride, N)
//
// Sums over a shape-(Df,S) array.
// No downsampling kernels are used -- the sum is purely "vertical".


template<int Df, typename T, int S, typename std::enable_if<(Df==0),int>::type = 0>
inline void _sum_strided(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, const T *in_w, int stride)
{ }

template<int Df, typename T, int S, typename std::enable_if<(Df>0),int>::type = 0>
inline void _sum_strided(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, const T *in_w, int stride)
{
    _sum_strided<Df-1> (wi_acc, w_acc, in_i, in_w, stride);
    
    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (in_i + (Df-1)*stride);
    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (in_w + (Df-1)*stride);

    wi_acc += wval * ival;
    w_acc += wval;
}


// ----------------------------------------------------------------------------------------------------
//
// _wi_downsampler_0d<T, S, Df0, DtX> ds0(Dt);
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
	_sum_strided<Df0> (wi_acc, w_acc, in_i + (P-1)*S, in_w + (P-1)*S, stride);
	
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
	
	for (int it = 0; it < Dt; it += S)
	    _sum_strided<Df0> (wi_acc, w_acc, in_i+(P-1)*Dt+it, in_w+(P-1)*Dt+it, stride);
	
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
// _wi_downsampler_1d_outbuf<T, S> out;
//
// _wi_downsampler_1d<T, S, DfX, DtX> ds1(Df, Dt);
// ds1.downsample_1d(out, nt_out, in_i, in_w, stride);


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


// 1D "outbuf" class.
// Important note!  In the "large Df" case, the _wi_downsampler will populate i_out, w_out
// with partially summed downsampled results.
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
//
// _wi_downsampler_1f<T, S, DtX> ds1(Df, Dt);
// ds1.downsample_1f(out, nfreq, i_in, w_in, stride);
//
// where 'out' is a class which defines
//   out.put(wival, wval, ifreq_ds)
//
// FIXME: not sure if this is fastest.


template<typename T, int S, int Dt, bool Dt_Large = (Dt > S)>
struct _wi_downsampler_1f;


// Case 1: "small" Dt.
template<typename T, int S, int Dt>
struct _wi_downsampler_1f<T, S, Dt, false>
{
    const int Df;

    _wi_downsampler_1f(int Df_, int Dt_) :
	Df(Df_)
    {
	if (__builtin_expect(Dt != Dt_, 0))
	    throw std::runtime_error("rf_kernels: internal error: \"small\" Dt mismatch in _wi_downsampler_1f()");
    }


    template<int D, typename std::enable_if<(D==0),int>::type = 0>
    inline void accumulate_row(simd_ntuple<T,S,D> &wi_acc, simd_ntuple<T,S,D> &w_acc, const T *i_in, const T *w_in) const
    { }

    template<int D, typename std::enable_if<(D>0),int>::type = 0>
    inline void accumulate_row(simd_ntuple<T,S,D> &wi_acc, simd_ntuple<T,S,D> &w_acc, const T *i_in, const T *w_in) const
    {
	accumulate_row(wi_acc.v, w_acc.v, i_in, w_in);

	simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_in + (D-1)*S);
	simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_in + (D-1)*S);

	wi_acc.x += wval * ival;
	w_acc.x += wval;
    }


    template<typename Tout>
    inline void downsample_1f(Tout &out, int nfreq_ds, const T *i_in, const T *w_in, int stride) const
    {
	for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++) {
	    simd_ntuple<T,S,Dt> wiacc;
	    simd_ntuple<T,S,Dt> wacc;

	    wiacc.loadu(i_in);
	    wacc.loadu(w_in);
	    wiacc *= wacc;

	    for (int i = 1; i < Df; i++)
		accumulate_row(wiacc, wacc, i_in + i*stride, w_in + i*stride);

	    out.put(simd_downsample(wiacc), simd_downsample(wacc), ifreq_ds);

	    i_in += Df * stride;
	    w_in += Df * stride;
	}
    }
};


// Case 2: "large" Dt.
template<typename T, int S, int DtX>
struct _wi_downsampler_1f<T, S, DtX, true>
{
    const int Df;
    const int Dt;

    _wi_downsampler_1f(int Df_, int Dt_) :
	Df(Df_),
	Dt(Dt_)
    {
	if (__builtin_expect((Dt <= S) || (Dt % S), 0))
	    throw std::runtime_error("rf_kernels: internal error: invalid \"large\" Dt in _wi_downsampler_1f()");
    }


    template<int D, typename std::enable_if<(D==0),int>::type = 0>
    inline void accumulate_row(simd_ntuple<T,S,D> &wi_acc, simd_ntuple<T,S,D> &w_acc, const T *i_in, const T *w_in) const
    { }

    template<int D, typename std::enable_if<(D>0),int>::type = 0>
    inline void accumulate_row(simd_ntuple<T,S,D> &wi_acc, simd_ntuple<T,S,D> &w_acc, const T *i_in, const T *w_in) const
    {
	accumulate_row(wi_acc.v, w_acc.v, i_in, w_in);

	for (int i = 0; i < Dt; i += S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_in + (D-1)*Dt + i);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_in + (D-1)*Dt + i);

	    wi_acc.x += wval * ival;
	    w_acc.x += wval;
	}
    }


    template<typename Tout>
    inline void downsample_1f(Tout &out, int nfreq_ds, const T *i_in, const T *w_in, int stride) const
    {
	for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++) {
	    simd_ntuple<T,S,S> wiacc;
	    simd_ntuple<T,S,S> wacc;

	    wiacc.setzero();
	    wacc.setzero();

	    for (int i = 0; i < Df; i++)
		accumulate_row(wiacc, wacc, i_in + i*stride, w_in + i*stride);

	    out.put(simd_downsample(wiacc), simd_downsample(wacc), ifreq_ds);

	    i_in += Df * stride;
	    w_in += Df * stride;
	}
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
