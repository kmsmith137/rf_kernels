#ifndef _RF_KERNELS_UPSAMPLE_INTERNALS_HPP
#define _RF_KERNELS_UPSAMPLE_INTERNALS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <simd_helpers/core.hpp>
#include <simd_helpers/simd_float32.hpp>
#include <simd_helpers/upsample.hpp>

#include "upsample.hpp"

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif

template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;
template<typename T, int S, int N> using simd_ntuple = simd_helpers::simd_ntuple<T,S,N>;
template<typename T, int S, int D> using simd_upsampler = simd_helpers::simd_upsampler<T,S,D>;


// -------------------------------------------------------------------------------------------------
//
// _mask_strided<N> (T *wp, simd_t<T,S> mask, int stride)


template<int N, typename T, int S, typename std::enable_if<(N==0),int>::type = 0>
inline void _mask_strided(T *wp, simd_t<T,S> mask, int stride)
{ }

template<int N, typename T, int S, typename std::enable_if<(N>0),int>::type = 0>
inline void _mask_strided(T *wp, simd_t<T,S> mask, int stride)
{
    _mask_strided<N-1> (wp, mask, stride);

    T *p = wp + (N-1)*stride;
    simd_t<T,S> x = simd_helpers::simd_load<T,S> (p);
    simd_helpers::simd_store(p, mask & x);
}


// -------------------------------------------------------------------------------------------------
//
// _weight_upsampler_0d<T, S, Df0, DtX> us0(Dt);
//
// simd_t<T,S> mask;
// us0.put_mask(w_out, stride, mask);


template<typename T, int S, int Df0, int DtX, bool Dt_Large = (DtX > S)>
struct _weight_upsampler_0d;


// Case 1: "Small" Dt.
template<typename T, int S, int Df0, int Dt>
struct _weight_upsampler_0d<T, S, Df0, Dt, false>
{
    _weight_upsampler_0d(int Dt_) 
    {
	if (__builtin_expect(Dt != Dt_, 0))
	    throw std::runtime_error("rf_kernels internal error: \"small\" Dt mismatch in _weight_upsampler_0d");
    }


    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_upsampler<T,S,Dt> &mask, int stride)
    { }

    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_upsampler<T,S,Dt> &mask, int stride)
    {   
	_put_mask_partial<P-1> (wp, mask, stride);
	_mask_strided<Df0> (wp + (P-1)*S, mask.template get<P-1>(), stride);
    }

    
    inline void put_mask(T *w_out, int stride, simd_t<T,S> mask)
    {
	_put_mask_partial<Dt> (w_out, simd_upsampler<T,S,Dt>(mask), stride);	
    }
};


// Case 2: "Large" Dt.
template<typename T, int S, int Df0, int DtX>
struct _weight_upsampler_0d<T, S, Df0, DtX, true>
{
    const int Dt;
 
    _weight_upsampler_0d(int Dt_) : 
	Dt(Dt_) 
    { 
	if (__builtin_expect((Dt_ <= S) || (Dt_ % S), 0))
	    throw std::runtime_error("rf_kernels internal error: bad \"large\" Dt in _weight_upsampler_0d");
    }

    
    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_upsampler<T,S,S> &mask, int stride)
    { }

    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_upsampler<T,S,S> &mask, int stride)
    {
	_put_mask_partial<P-1> (wp, mask, stride);

	simd_t<T,S> m = mask.template get<P-1>();

	for (int i = 0; i < Dt; i += S)
	    _mask_strided<Df0> (wp + (P-1)*Dt + i, m, stride);
    }

    
    inline void put_mask(T *w_out, int stride, simd_t<T,S> mask)
    {
	_put_mask_partial<S> (w_out, simd_upsampler<T,S,S>(mask), stride);
    }
};


// -------------------------------------------------------------------------------------------------
//
// _weight_upsampler_0f<T, S, DfX, DtX> us0(Df, Dt)
//
// simd_t<T,S>
// us0.put_mask(w_out, stride, mask)
//
// Note: _weight_upsampler_0e is a helper class which masks one row.
//
// FIXME: might want to consider defining a "medium" _weight_upsampler_0e,
//  if timings slow down at Dt=16.


template<typename T, int S, int DtX, bool Dt_Large = (DtX > S)>
struct _weight_upsampler_0e;

template<typename T, int S, int DfX, int DtX, bool Df_Large = (DfX > S)>
struct _weight_upsampler_0f;


// Case 1: "Small" Dt.
template<typename T, int S, int Dt>
struct _weight_upsampler_0e<T, S, Dt, false>
{
    static constexpr int U = Dt;
    
    _weight_upsampler_0e(int Dt_)
    {
	if (__builtin_expect(Dt != Dt_, 0))
	    throw std::runtime_error("rf_kernels: internal error: \"small\" Dt mismatch in _weight_upsampler_0e");
    }
    
    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_ntuple<T,S,Dt> &mask)
    { }
    
    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_ntuple<T,S,Dt> &mask)
    {
	_put_mask_partial<P-1> (wp, mask);

	T *p = wp + (P-1)*S;
	simd_t<T,S> m = mask.template extract<P-1> ();
	simd_t<T,S> x = simd_helpers::simd_load<T,S> (p);
	simd_helpers::simd_store(p, m & x);
    }
    
    inline void put_mask(T *wp, const simd_ntuple<T,S,Dt> &mask)
    {
	_put_mask_partial<Dt> (wp, mask);
    }
};


// Case 2: "Large" Dt.
template<typename T, int S, int DtX>
struct _weight_upsampler_0e<T, S, DtX, true>
{
    static constexpr int U = S;
    const int Dt;

    _weight_upsampler_0e(int Dt_) :
	Dt(Dt_)
    {
	if (__builtin_expect((Dt <= S) || (Dt % S), 0))
	    throw std::runtime_error("rf_kernels: internal error: invalid \"large\" Dt in _weight_upsampler_0e");
    }

    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_ntuple<T,S,S> &mask)
    { }
    
    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_ntuple<T,S,S> &mask)
    {
	_put_mask_partial<P-1> (wp, mask);

	T *p = wp + (P-1)*Dt;
	simd_t<T,S> m = mask.template extract<P-1> ();

	for (int i = 0; i < Dt; i += S) {
	    simd_t<T,S> x = simd_helpers::simd_load<T,S> (p+i);
	    simd_helpers::simd_store(p+i, m & x);
	}
    }
    
    inline void put_mask(T *wp, const simd_ntuple<T,S,S> &mask)
    {
	_put_mask_partial<S> (wp, mask);
    }
};


// Case 1: "Small" Df.
template<typename T, int S, int Df, int DtX>
struct _weight_upsampler_0f<T, S, Df, DtX, false>
{
    _weight_upsampler_0e<T,S,DtX> _us;
    
    _weight_upsampler_0f(int Df_, int Dt) :
	_us(Dt)
    {
	if (__builtin_expect(Df != Df_, 0))
	    throw std::runtime_error("rf_kernels: internal error: \"small\" Df mismatch in _weight_upsampler_0f");
    }

    
    template<int P, int U, typename std::enable_if<(P==0),int>::type = 0>
    inline void _put_mask_partial(T *wp, int stride, const simd_ntuple<T,S,U> &mask)
    { }
    
    template<int P, int U, typename std::enable_if<(P>0),int>::type = 0>
    inline void _put_mask_partial(T *wp, int stride, const simd_ntuple<T,S,U> &mask)
    {
	_put_mask_partial<P-1> (wp, stride, mask);
	_us.put_mask(wp + (P-1)*stride, mask);
    }

    
    inline void put_mask(T *wp, int stride, simd_t<T,S> mask)
    {
	static constexpr int U = decltype(_us)::U;
	
	simd_ntuple<T,S,U> m;
	simd_upsample(m, mask);
	
	_put_mask_partial<Df> (wp, stride, m);
    }
};


// Case 2: "Large" Df.
template<typename T, int S, int DfX, int DtX>
struct _weight_upsampler_0f<T, S, DfX, DtX, true>
{
    const int Df;
    _weight_upsampler_0e<T,S,DtX> _us;
    
    _weight_upsampler_0f(int Df_, int Dt) :
	Df(Df_),
	_us(Dt)
    {
	if (__builtin_expect((Df <= S) || (Df % S), 0))
	    throw std::runtime_error("rf_kernels: internal error: invalid \"large\" Df in _weight_upsampler_0f");
    }

    
    template<int P, int U, typename std::enable_if<(P==0),int>::type = 0>
    inline void _put_mask_partial(T *wp, int stride, const simd_ntuple<T,S,U> &mask)
    { }
    
    template<int P, int U, typename std::enable_if<(P>0),int>::type = 0>
    inline void _put_mask_partial(T *wp, int stride, const simd_ntuple<T,S,U> &mask)
    {
	_put_mask_partial<P-1> (wp, stride, mask);
	_us.put_mask(wp + (P-1)*stride, mask);
    }

    
    inline void put_mask(T *wp, int stride, simd_t<T,S> mask)
    {
	static constexpr int U = decltype(_us)::U;
	
	simd_ntuple<T,S,U> m;
	simd_upsample(m, mask);

	for (int i = 0; i < Df; i += S)
	    _put_mask_partial<S> (wp + i*stride, stride, m);
    }
};



// -------------------------------------------------------------------------------------------------
//
// Bottom-line kernel.


template<typename T, int S, int Df0, int DtX>
inline void kernel_upsample_weights(const weight_upsampler *wp, int nfreq_in, int nt_in, T *w_out, int ostride, const T *w_in, int istride, T w_cutoff_)
{
    const int Df = wp->Df;
    const int Dt = wp->Dt;
    const simd_t<T,S> w_cutoff = w_cutoff_;

    if (__builtin_expect((Df <= 0) || (Df % Df0), 0))
	throw std::runtime_error("rf_kernels: internal error: bad (Df,Df0) pair in kernel_upsample_weights");

    _weight_upsampler_0d<T, S, Df0, DtX> us0(Dt);

    for (int ifreq_in = 0; ifreq_in < nfreq_in; ifreq_in++) {
	const T *w_in2 = w_in + ifreq_in * istride;

	for (int ifreq_out = ifreq_in*Df; ifreq_out < (ifreq_in+1)*Df; ifreq_out += Df0) {
	    T *w_out2 = w_out + ifreq_out * ostride;

	    for (int it = 0; it < nt_in; it += S) {
		simd_t<T,S> w = simd_helpers::simd_load<T,S> (w_in2 + it);
		simd_t<T,S> mask = (w > w_cutoff);
		us0.put_mask(w_out2 + it*Dt, ostride, mask);
	    }
	}
    }    
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_UPSAMPLE_INTERNALS_HPP
