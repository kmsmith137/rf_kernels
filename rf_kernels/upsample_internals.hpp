#ifndef _RF_KERNELS_UPSAMPLE_INTERNALS_HPP
#define _RF_KERNELS_UPSAMPLE_INTERNALS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <simd_helpers/core.hpp>
#include <simd_helpers/simd_float32.hpp>
#include <simd_helpers/upsample.hpp>

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif

template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;
template<typename T, int S, int D> using simd_upsampler = simd_helpers::simd_upsampler<T,S,D>;


// -------------------------------------------------------------------------------------------------
//
// _upsample_weights_0a<Df> (float *wp, simd_t<float,8> mask, int stride)
// _upsample_weights_0a<Df> (float *wp, simd_t<float,8>, int stride, int N)
//
// These apply the mask to a shape-(Df,S) and shape-(Df,N) array respectively.
// In the latter case, N must be divisible by the simd size S!


template<int Df, typename std::enable_if<(Df==0),int>::type = 0>
inline void _upsample_weights_0a(float *wp, simd_t<float,8> mask, int stride)
{ }

template<int Df, typename std::enable_if<(Df>0),int>::type = 0>
inline void _upsample_weights_0a(float *wp, simd_t<float,8> mask, int stride)
{
    _upsample_weights_0a<Df-1> (wp, mask, stride);

    float *p = wp + (Df-1)*stride;
    simd_t<float,8> x = simd_helpers::simd_load<float,8> (p);
    
    simd_helpers::simd_store(p, mask & x);
}

template<int Df, typename std::enable_if<(Df>0),int>::type = 0>
inline void _upsample_weights_0a(float *wp, simd_t<float,8> mask, int stride, int N)
{
    for (int i = 0; i < N; i += 8)
	_upsample_weights_0a<Df> (wp+i, mask, stride);
}


// ------------------------------------------------------------------------------------------------
//
// _upsample_weights_0b<Df,Dt,P> (float *p, const simd_upsampler<Dt> &mask, int stride)
// _upsample_weights_0b<Df,P> (float *p, const simd_upsampler<S> &mask, int stride, int Dt)
//
// These apply the mask to a shape-(Df,Dt*P) array, where P <= S.
// They are "partial" versions of _upsample_weights_0d(), to be defined next.


template<int Df, int Dt, int P, typename std::enable_if<(P==0),int>::type = 0>
inline void _upsample_weights_0b(float *wp, const simd_upsampler<float,8,Dt> &mask, int stride)
{ }

template<int Df, int Dt, int P, typename std::enable_if<(P>0),int>::type = 0>
inline void _upsample_weights_0b(float *wp, const simd_upsampler<float,8,Dt> &mask, int stride)
{   
    constexpr int Q = (P-1)*8;
    _upsample_weights_0b<Df,Dt,P-1> (wp, mask, stride);
    _upsample_weights_0a<Df> (wp+Q, mask.template get<P-1>(), stride);
}

template<int Df, int P, typename std::enable_if<(P==0),int>::type = 0>
inline void _upsample_weights_0b(float *wp, const simd_upsampler<float,8,8> &mask, int stride, int Dt)
{ }

template<int Df, int P, typename std::enable_if<(P>0),int>::type = 0>
inline void _upsample_weights_0b(float *wp, const simd_upsampler<float,8,8> &mask, int stride, int Dt)
{
    _upsample_weights_0b<Df,P-1> (wp, mask, stride, Dt);
    _upsample_weights_0a<Df> (wp+(P-1)*Dt, mask.template get<P-1>(), stride, Dt);
}


// -------------------------------------------------------------------------------------------------
//
// _upsample_weights_0d<Df,Dt> (float *p, simd_t<float,8> mask, int stride)
// _upsample_weights_0d<Df> (float *p, simd_t<float,8> mask, int stride, int Dt)
//
// These apply the mask to a shape-(Df,Dt*S) array.

template<int Df, int Dt>
inline void _upsample_weights_0d(float *wp, simd_t<float,8> mask, int stride)
{
    _upsample_weights_0b<Df,Dt,Dt> (wp, simd_upsampler<float,8,Dt>(mask), stride);
}

template<int Df>
inline void _upsample_weights_0d(float *wp, simd_t<float,8> mask, int stride, int Dt)
{
    _upsample_weights_0b<Df,8> (wp, simd_upsampler<float,8,8>(mask), stride, Dt);
}


// -------------------------------------------------------------------------------------------------
//
// _upsample_weights_1d<Df,Dt> (int nt_in, float *out, int ostride, const float *in, __m256 w_cutoff)
// _upsample_weights_1d<Df>    (int nt_in, float *out, int ostride, const float *in, __m256 w_cutoff, int Dt)
//
// These operate on an 1D input array of length nt_in, and an output array of shape (Df,Dt*nt_in).
// In the first case, Dt is a compile-time parameter, and in the second case it is a runtime parameter
// In the second case, Dt must be divisible by the simd size S=8!
//
// Caller must check that nt_in is divisible by the simd_size S=8.


template<int Df, int Dt>
inline void _upsample_weights_1d(int nt_in, float *out, int ostride, const float *in, __m256 w_cutoff)
{
    for (int it = 0; it < nt_in; it += 8) {
	// FIXME change __m256 to simd_t<float,8>, once smask stuff is straightened out
	__m256 w = _mm256_loadu_ps(in+it);
	__m256 mask = _mm256_cmp_ps(w, w_cutoff, _CMP_GT_OQ);
	_upsample_weights_0d<Df,Dt> (out+it*Dt, mask, ostride);
    }
}

template<int Df>
inline void _upsample_weights_1d(int nt_in, float *out, int ostride, const float *in, __m256 w_cutoff, int Dt)
{
    for (int it = 0; it < nt_in; it += 8) {
	// FIXME change __m256 to simd_t<float,8>, once smask stuff is straightened out
	__m256 w = _mm256_loadu_ps(in+it);
	__m256 mask = _mm256_cmp_ps(w, w_cutoff, _CMP_GT_OQ);
	_upsample_weights_0d<Df> (out+it*Dt, mask, ostride, Dt);
    }
}


// -------------------------------------------------------------------------------------------------
//
// kernel_upsample_weights_Df_Dt()
// kernel_upsample_weights_Df()
// kernel_upsample_weights_Dt()
//
// Caller must check all arguments, including these checks:
//   (nt_in % 8) == 0
//   Df == Df_
//   Dt == Dt_
//
// The (Df_, Dt_) arguments are superfluous!


template<int Df, int Dt>
inline void kernel_upsample_weights_Df_Dt(int nfreq_in, int nt_in, float *dst, int dstride, const float *src, int sstride, float w_cutoff_, int Df_, int Dt_)
{
    __m256 w_cutoff = _mm256_set1_ps(w_cutoff_);

    for (int ifreq = 0; ifreq < nfreq_in; ifreq++)
	_upsample_weights_1d<Df,Dt> (nt_in, dst + ifreq*Df*dstride, dstride, src + ifreq*sstride, w_cutoff);
}


template<int Df>
inline void kernel_upsample_weights_Df(int nfreq_in, int nt_in, float *dst, int dstride, const float *src, int sstride, float w_cutoff_, int Df_, int Dt)
{
    __m256 w_cutoff = _mm256_set1_ps(w_cutoff_);

    for (int ifreq = 0; ifreq < nfreq_in; ifreq++)
	_upsample_weights_1d<Df> (nt_in, dst + ifreq*Df*dstride, dstride, src + ifreq*sstride, w_cutoff, Dt);
}


template<int Dt>
inline void kernel_upsample_weights_Dt(int nfreq_in, int nt_in, float *dst, int dstride, const float *src, int sstride, float w_cutoff_, int Df, int Dt_)
{
    __m256 w_cutoff = _mm256_set1_ps(w_cutoff_);

    for (int ifreq_in = 0; ifreq_in < nfreq_in; ifreq_in++)
	for (int ifreq_out = ifreq_in * Df; ifreq_out < (ifreq_in+1) * Df; ifreq_out += 8)
	    _upsample_weights_1d<8,Dt> (nt_in, dst + ifreq_out*dstride, dstride, src + ifreq_in*sstride, w_cutoff);
}


// Not a template
inline void kernel_upsample_weights(int nfreq_in, int nt_in, float *dst, int dstride, const float *src, int sstride, float w_cutoff_, int Df, int Dt)
{
    __m256 w_cutoff = _mm256_set1_ps(w_cutoff_);

    for (int ifreq_in = 0; ifreq_in < nfreq_in; ifreq_in++)
	for (int ifreq_out = ifreq_in * Df; ifreq_out < (ifreq_in+1) * Df; ifreq_out += 8)
	    _upsample_weights_1d<8> (nt_in, dst + ifreq_out*dstride, dstride, src + ifreq_in*sstride, w_cutoff, Dt);
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_UPSAMPLE_INTERNALS_HPP
