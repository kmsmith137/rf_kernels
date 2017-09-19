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
// _sum_strided<Df> (wi_acc, w_acc, intensity, istride, weights, wstride)
// _sum_strided<Df> (wi_acc, w_acc, intensity, istride, weights, wstride, N)
//
// Sums over a shape-(Df,S) array.
// No downsampling kernels are used -- the sum is purely "vertical".


template<int Df, typename T, int S, typename std::enable_if<(Df==0),int>::type = 0>
inline void _sum_strided(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, int istride, const T *in_w, int wstride)
{ }

template<int Df, typename T, int S, typename std::enable_if<(Df>0),int>::type = 0>
inline void _sum_strided(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, int istride, const T *in_w, int wstride)
{
    _sum_strided<Df-1> (wi_acc, w_acc, in_i, istride, in_w, wstride);
    
    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (in_i + (Df-1)*istride);
    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (in_w + (Df-1)*wstride);

    wi_acc += wval * ival;
    w_acc += wval;
}


// ----------------------------------------------------------------------------------------------------
//
// _wi_downsampler_0d<T, S, Df0, DtX> ds0(Dt);
//
// simd_t<T,S> wi_out, w_out;
// ds0.get(wi_out, w_out, i_in, istride, w_in, wstride);
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
    inline void _get_partial(simd_downsampler<T,S,Dt> &wi_ds, simd_downsampler<T,S,Dt> &w_ds, const T *in_i, int istride, const T *in_w, int wstride) const
    { }

    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,Dt> &wi_ds, simd_downsampler<T,S,Dt> &w_ds, const T *in_i, int istride, const T *in_w, int wstride) const
    {
	_get_partial<P-1> (wi_ds, w_ds, in_i, istride, in_w, wstride);
	
	simd_t<T,S> wi_acc = simd_t<T,S>::zero();
	simd_t<T,S> w_acc = simd_t<T,S>::zero();
	_sum_strided<Df0> (wi_acc, w_acc, in_i + (P-1)*S, istride, in_w + (P-1)*S, wstride);
	
	wi_ds.template put<P-1> (wi_acc);
	w_ds.template put<P-1> (w_acc);
    }

    
    inline void get(simd_t<T,S> &wi_out, simd_t<T,S> &w_out, const T *i_in, int istride, const T *w_in, int wstride) const
    {
	simd_downsampler<T,S,Dt> wi_ds, w_ds;
	_get_partial<Dt> (wi_ds, w_ds, i_in, istride, w_in, wstride);

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
    inline void _get_partial(simd_downsampler<T,S,S> &wi_ds, simd_downsampler<T,S,S> &w_ds, const T *in_i, int istride, const T *in_w, int wstride) const
    { }
    
    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,S> &wi_ds, simd_downsampler<T,S,S> &w_ds, const T *in_i, int istride, const T *in_w, int wstride) const
    {
	_get_partial<P-1> (wi_ds, w_ds, in_i, istride, in_w, wstride);
	
	simd_t<T,S> wi_acc = simd_t<T,S>::zero();
	simd_t<T,S> w_acc = simd_t<T,S>::zero();
	
	for (int it = 0; it < Dt; it += S)
	    _sum_strided<Df0> (wi_acc, w_acc, in_i+(P-1)*Dt+it, istride, in_w+(P-1)*Dt+it, wstride);
	
	wi_ds.template put<P-1> (wi_acc);
	w_ds.template put<P-1> (w_acc);    
    }

    
    inline void get(simd_t<T,S> &wi_out, simd_t<T,S> &w_out, const T *i_in, int istride, const T *w_in, int wstride) const
    {
	simd_downsampler<T,S,S> wi_ds, w_ds;
	_get_partial<S> (wi_ds, w_ds, i_in, istride, w_in, wstride);
	
	wi_out += wi_ds.get();
	w_out += w_ds.get();
    }
};


// -------------------------------------------------------------------------------------------------
//
// _wi_downsampler_1d<T, S, DfX, DtX> ds1(Df, Dt);
// ds1.downsample_1d(out, nt_out, in_i, istride, in_w, wstride);


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

    template<typename Tacc>
    inline void downsample_1d(Tacc &acc, int nt_out, const T *i_in, int istride, const T *w_in, int wstride, T *i_out, T *w_out)
    {
	const int Dt = ds0.get_Dt();
	const simd_t<T,S> zero = 0;
	const simd_t<T,S> one = 1;

	for (int it = 0; it < nt_out; it += S) {
	    simd_t<T,S> wival = simd_t<T,S>::zero();
	    simd_t<T,S> wval = simd_t<T,S>::zero();
	    ds0.get(wival, wval, i_in + it*Dt, istride, w_in + it*Dt, wstride);
	    
	    // FIXME revisit after smask cleanup.
	    simd_t<T,S> ival = wival / blendv(wval.compare_gt(zero), wval, one);
	    ival.storeu(i_out + it);
	    wval.storeu(w_out + it);	

	    acc.ds_put(ival, wval, wival);
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

    template<typename Tacc>
    inline void downsample_1d(Tacc &acc, int nt_out, const T *i_in, int in_istride, const T *w_in, int in_wstride, T *i_out, T *w_out)
    {
	const int Dt = ds0.get_Dt();
	const simd_t<T,S> zero = 0;
	const simd_t<T,S> one = 1;
	
	simd_t<T,S> wival;
	simd_t<T,S> wval;

	// First pass
	for (int it = 0; it < nt_out; it += S) {
	    wival = simd_t<T,S>::zero();
	    wval = simd_t<T,S>::zero();
	    ds0.get(wival, wval, i_in + it*Dt, in_istride, w_in + it*Dt, in_wstride);
	    
	    wival.storeu(i_out + it);
	    wval.storeu(w_out + it);
	}

	i_in += S * in_istride;
	w_in += S * in_wstride;
	
	// Middle passes
	for (int i = S; i < (Df-S); i += S) {
	    for (int it = 0; it < nt_out; it += S) {
		wival.loadu(i_out + it);
		wval.loadu(w_out + it);		
		ds0.get(wival, wval, i_in + it*Dt, in_istride, w_in + it*Dt, in_wstride);
		
		wival.storeu(i_out + it);
		wval.storeu(w_out + it);
	    }
	    
	    i_in += S * in_istride;
	    w_in += S * in_wstride;
	}

	// Last pass
	for (int it = 0; it < nt_out; it += S) {
	    wival.loadu(i_out + it);
	    wval.loadu(w_out + it);
	    ds0.get(wival, wval, i_in + it*Dt, in_istride, w_in + it*Dt, in_wstride);
	    
	    // FIXME revisit after smask cleanup.
	    simd_t<T,S> ival = wival / blendv(wval.compare_gt(zero), wval, one);
	    ival.storeu(i_out + it);
	    wval.storeu(w_out + it);	

	    acc.ds_put(ival, wval, wival);
	}
    }
};


// -------------------------------------------------------------------------------------------------
//
// _wi_downsampler_0f<T, S, DfX, DtX> ds1(Df, Dt);
// ds1.downsample_0f(out, nfreq, i_in, istride, w_in, wstride);
//
// where 'out' is a class which defines
//   out.put(wival, wval, ifreq_ds)
//
// Note: _wi_downsampler_0e is a helper class which processes one row.
//
// FIXME: not sure if this is fastest.
//
// FIXME: might want to consider defining a "medium" _weight_upsampler_0e,
//  if timings slow down at Dt=16.


template<typename T, int S, int DtX, bool Dt_Large = (DtX > S)>
struct _wi_downsampler_0e;

template<typename T, int S, int DfX, int DtX, bool Df_Large = (DfX > S)>
struct _wi_downsampler_0f;


// Case 1: "small" Dt.
template<typename T, int S, int Dt>
struct _wi_downsampler_0e<T, S, Dt, false>
{
    static constexpr int D = Dt;
	
    _wi_downsampler_0e(int Dt_)
    {
	if (__builtin_expect(Dt != Dt_, 0))
	    throw std::runtime_error("rf_kernels: internal error: \"small\" Dt mismatch in _wi_downsampler_0e()");
    }


    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _accumulate_row(simd_ntuple<T,S,P> &wi_acc, simd_ntuple<T,S,P> &w_acc, const T *i_in, const T *w_in)
    { }

    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _accumulate_row(simd_ntuple<T,S,P> &wi_acc, simd_ntuple<T,S,P> &w_acc, const T *i_in, const T *w_in)
    {
	_accumulate_row(wi_acc.v, w_acc.v, i_in, w_in);

	simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_in + (P-1)*S);
	simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_in + (P-1)*S);

	wi_acc.x += wval * ival;
	w_acc.x += wval;
    }

    inline void accumulate_row(simd_ntuple<T,S,Dt> &wi_acc, simd_ntuple<T,S,Dt> &w_acc, const T *i_in, const T *w_in)
    {
	_accumulate_row(wi_acc, w_acc, i_in, w_in);
    }
};


// Case 2: "large" Dt
template<typename T, int S, int DtX>
struct _wi_downsampler_0e<T, S, DtX, true>
{
    static constexpr int D = S;
    const int Dt;

    _wi_downsampler_0e(int Dt_) :
	Dt(Dt_)
    {
	if (__builtin_expect((Dt <= S) || (Dt % S), 0))
	    throw std::runtime_error("rf_kernels: internal_error: invalid \"large\" Dt in _wi_downsampler_0e");
    }


    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _accumulate_row(simd_ntuple<T,S,P> &wi_acc, simd_ntuple<T,S,P> &w_acc, const T *i_in, const T *w_in)
    { }

    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _accumulate_row(simd_ntuple<T,S,P> &wi_acc, simd_ntuple<T,S,P> &w_acc, const T *i_in, const T *w_in)
    {
	_accumulate_row(wi_acc.v, w_acc.v, i_in, w_in);

	for (int i = 0; i < Dt; i += S) {
	    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (i_in + (P-1)*Dt + i);
	    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (w_in + (P-1)*Dt + i);
	    
	    wi_acc.x += wval * ival;
	    w_acc.x += wval;
	}
    }

    inline void accumulate_row(simd_ntuple<T,S,S> &wi_acc, simd_ntuple<T,S,S> &w_acc, const T *i_in, const T *w_in)
    {
	_accumulate_row(wi_acc, w_acc, i_in, w_in);
    }
};


// Case 1: "small" Df
template<typename T, int S, int Df, int DtX>
struct _wi_downsampler_0f<T, S, Df, DtX, false>
{
    _wi_downsampler_0e<T, S, DtX> _ds;
    
    _wi_downsampler_0f(int Df_, int Dt) :
	_ds(Dt)
    {
	if (__builtin_expect(Df != Df_, 0))
	    throw std::runtime_error("rf_kernels: internal error: \"small\" Df mismatch in _wi_downsampler_0f()");
    }

    inline constexpr int get_Df() const { return Df; }
    

    template<int P, int D, typename std::enable_if<(P==0),int>::type = 0>
    inline void _accumulate_rows(simd_ntuple<T,S,D> &wi_acc, simd_ntuple<T,S,D> &w_acc, const T *i_in, int istride, const T *w_in, int wstride)
    { }

    template<int P, int D, typename std::enable_if<(P>0),int>::type = 0>
    inline void _accumulate_rows(simd_ntuple<T,S,D> &wi_acc, simd_ntuple<T,S,D> &w_acc, const T *i_in, int istride, const T *w_in, int wstride)
    {
	_accumulate_rows<P-1> (wi_acc, w_acc, i_in, istride, w_in, wstride);
	_ds.accumulate_row(wi_acc, w_acc, i_in + (P-1)*istride, w_in + (P-1)*wstride);
    }


    inline void get(simd_t<T,S> &wival, simd_t<T,S> &wval, const T *i_in, int istride, const T *w_in, int wstride)
    {
	constexpr int D = decltype(_ds)::D;

	simd_ntuple<T,S,D> wi_acc, w_acc;
	wi_acc.setzero();
	w_acc.setzero();

	_accumulate_rows<Df> (wi_acc, w_acc, i_in, istride, w_in, wstride);

	wival = simd_downsample(wi_acc);
	wval = simd_downsample(w_acc);
    }
};


// Case 2: "large Df"
template<typename T, int S, int DfX, int DtX>
struct _wi_downsampler_0f<T, S, DfX, DtX, true>
{
    const int Df;
    _wi_downsampler_0e<T, S, DtX> _ds;
    
    _wi_downsampler_0f(int Df_, int Dt) :
	Df(Df_),
	_ds(Dt)
    {
	if (__builtin_expect((Df <= S) || (Df % S), 0))
	    throw std::runtime_error("rf_kernels: internal error: invalid \"large\" Df in _wi_downsampler_0f()");
    }

    
    inline int get_Df() const { return Df; }

    
    template<int P, int D, typename std::enable_if<(P==0),int>::type = 0>
    inline void _accumulate_rows(simd_ntuple<T,S,D> &wi_acc, simd_ntuple<T,S,D> &w_acc, const T *i_in, int istride, const T *w_in, int wstride)
    { }

    template<int P, int D, typename std::enable_if<(P>0),int>::type = 0>
    inline void _accumulate_rows(simd_ntuple<T,S,D> &wi_acc, simd_ntuple<T,S,D> &w_acc, const T *i_in, int istride, const T *w_in, int wstride)
    {
	_accumulate_rows<P-1> (wi_acc, w_acc, i_in, istride, w_in, wstride);
	_ds.accumulate_row(wi_acc, w_acc, i_in + (P-1)*istride, w_in + (P-1)*wstride);
    }

    
    inline void get(simd_t<T,S> &wival, simd_t<T,S> &wval, const T *i_in, int istride, const T *w_in, int wstride)
    {
	constexpr int D = decltype(_ds)::D;

	simd_ntuple<T,S,D> wi_acc, w_acc;
	wi_acc.setzero();
	w_acc.setzero();

	for (int i = 0; i < Df; i += S)
	    _accumulate_rows<S> (wi_acc, w_acc, i_in + i*istride, istride, w_in + i*wstride, wstride);

	wival = simd_downsample(wi_acc);
	wval = simd_downsample(w_acc);
    }
};



// -------------------------------------------------------------------------------------------------


template<typename T, int S, int DfX, int DtX>
struct _wi_downsampler_1f {
    _wi_downsampler_0f<T,S,DfX,DtX> ds0;

    _wi_downsampler_1f(int Df, int Dt) :
	ds0(Df, Dt)
    { }


    template<typename Tacc>
    inline void downsample_1f(Tacc &acc, int nfreq_ds, const T *i_in, int in_istride, const T *w_in, int in_wstride, T *i_out, T *w_out)
    {
	const int Df = ds0.get_Df();
	const simd_t<T,S> zero = 0;
	const simd_t<T,S> one = 1;

	for (int ifreq_ds = 0; ifreq_ds < nfreq_ds; ifreq_ds++) {
	    simd_t<T,S> wival, wval;
	    
	    ds0.get(wival, wval,
		    i_in + ifreq_ds*Df*in_istride, in_istride,
		    w_in + ifreq_ds*Df*in_wstride, in_wstride);
	    
	    // FIXME revisit after smask cleanup.
	    simd_t<T,S> ival = wival / blendv(wval.compare_gt(zero), wval, one);
	    ival.storeu(i_out + ifreq_ds*S);
	    wval.storeu(w_out + ifreq_ds*S);	
	    
	    acc.ds_put(ival, wval, wival);
	}
    }
};


// -------------------------------------------------------------------------------------------------


template<typename T, int S>
struct _dummy_wi_ds_accumulator
{
    inline void ds_put(simd_t<T,S> ival, simd_t<T,S> wval, simd_t<T,S> wival) { }
};


template<typename T, int S, int DfX, int DtX, typename std::enable_if<((DfX > 1) || (DtX > 1)),int>::type = 0>
inline void kernel_wi_downsample(const wi_downsampler *dp, int nfreq_out, int nt_out, T *out_i, int out_istride,
				 T *out_w, int out_wstride, const T *in_i, int in_istride, const T *in_w, int in_wstride)
{
    const int Df = dp->Df;
    const int Dt = dp->Dt;
    
    _dummy_wi_ds_accumulator<T,S> acc;
    _wi_downsampler_1d<T, S, DfX, DtX> ds1(Df, Dt);
    
    for (int ifreq = 0; ifreq < nfreq_out; ifreq++) {
	ds1.downsample_1d(acc, nt_out,
			  in_i + ifreq * Df * in_istride, in_istride,
			  in_w + ifreq * Df * in_wstride, in_wstride,
			  out_i + ifreq * out_istride,
			  out_w + ifreq * out_wstride);
    }
}


// Special case (Df,Dt)=(1,1).
template<typename T, int S, int DfX, int DtX, typename std::enable_if<((DfX == 1) && (DtX == 1)),int>::type = 0>
inline void kernel_wi_downsample(const wi_downsampler *dp, int nfreq_out, int nt_out, T *out_i, int out_istride,
				 T *out_w, int out_wstride, const T *in_i, int in_istride, const T *in_w, int in_wstride)
{
    for (int ifreq = 0; ifreq < nfreq_out; ifreq++)
	memcpy(out_i + ifreq*out_istride, in_i + ifreq*in_istride, nt_out * sizeof(T));
    
    for (int ifreq = 0; ifreq < nfreq_out; ifreq++)
	memcpy(out_w + ifreq*out_wstride, in_w + ifreq*in_wstride, nt_out * sizeof(T));
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_DOWNSAMPLE_INTERNALS_HPP
