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
// _upsample_weights_0d<Df,Dt> (T *p, simd_t<T,S> mask, int stride)
// _upsample_weights_0d<Df> (T *p, simd_t<T,S> mask, int stride, int Dt)
//
// These apply the mask to a shape-(Df,Dt*S) array.


template<typename T_, int S_, int Df, int Dt>
struct _weight_upsampler_0d_Dtsm {
    using T = T_;
    static constexpr int S = S_;

    inline constexpr int get_Df() const { return Df; }
    inline constexpr int get_Dt() const { return Dt; }

    
    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_upsampler<T,S,Dt> &mask, int stride)
    { }

    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_upsampler<T,S,Dt> &mask, int stride)
    {   
	constexpr int Q = (P-1)*S;
	_put_mask_partial<P-1> (wp, mask, stride);
	_mask_strided<Df> (wp+Q, mask.template get<P-1>(), stride);
    }

    
    inline void put_mask(T *w_out, int stride, simd_t<T,S> mask)
    {
	_put_mask_partial<Dt> (w_out, simd_upsampler<T,S,Dt>(mask), stride);	
    }
};


template<typename T_, int S_, int Df>
struct _weight_upsampler_0d_Dtlg {
    using T = T_;
    static constexpr int S = S_;
    
    const int Dt;
    
    _weight_upsampler_0d_Dtlg(int Dt_) : Dt(Dt_) { }

    inline constexpr int get_Df() const { return Df; }
    inline int get_Dt() const { return Dt; }

    
    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_upsampler<T,S,S> &mask, int stride)
    { }

    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _put_mask_partial(T *wp, const simd_upsampler<T,S,S> &mask, int stride)
    {
	_put_mask_partial<P-1> (wp, mask, stride);

	simd_t<T,S> m = mask.template get<P-1>();

	for (int i = 0; i < Dt; i += S)
	    _mask_strided<Df> (wp + (P-1)*Dt + i, m, stride);
    }

    
    inline void put_mask(T *w_out, int stride, simd_t<T,S> mask)
    {
	_put_mask_partial<S> (w_out, simd_upsampler<T,S,S>(mask), stride);
    }
};


// -------------------------------------------------------------------------------------------------
//
// _upsample_weights_2d()


template<typename Tus0, typename T = typename Tus0::T, int S = Tus0::S>
inline void _upsample_weights_2d(Tus0 &us0, int Df, int nfreq_in, int nt_in, T *w_out, int ostride, const T *w_in, int istride, T w_cutoff_)
{
    const int Df0 = us0.get_Df();
    const int Dt = us0.get_Dt();
    simd_t<T,S> w_cutoff = w_cutoff_;

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


// -------------------------------------------------------------------------------------------------
//
// Low-level kernels for rf_kernels::weight_upsampler.


template<typename T, int S, int Df, int Dt>
inline void kernel_upsample_weights_Dtsm(const weight_upsampler *wp, int nfreq_in, int nt_in, T *dst, int dstride, const T *src, int sstride, T w_cutoff)
{
    _weight_upsampler_0d_Dtsm<T,S,Df,Dt> us0;
    _upsample_weights_2d(us0, wp->Df, nfreq_in, nt_in, dst, dstride, src, sstride, w_cutoff);
}


template<typename T, int S, int Df>
inline void kernel_upsample_weights_Dtlg(const weight_upsampler *wp, int nfreq_in, int nt_in, T *dst, int dstride, const T *src, int sstride, T w_cutoff)
{
    _weight_upsampler_0d_Dtlg<T,S,Df> us0(wp->Dt);
    _upsample_weights_2d(us0, wp->Df, nfreq_in, nt_in, dst, dstride, src, sstride, w_cutoff);
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_UPSAMPLE_INTERNALS_HPP
