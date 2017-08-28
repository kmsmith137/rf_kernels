#ifndef _RF_KERNELS_UPSAMPLE_INTERNALS_HPP
#define _RF_KERNELS_UPSAMPLE_INTERNALS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <simd_helpers/core.hpp>
#include <simd_helpers/simd_float32.hpp>
#include <simd_helpers/simd_ntuple.hpp>

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif

template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;
template<typename T, int S, int D> using simd_ntuple = simd_helpers::simd_ntuple<T,S,D>;


// -------------------------------------------------------------------------------------------------
//
// struct simd_upsampler<Dt>
//
// Eventually, this code can be generalized to a pair <T,S> and moved to simd_helpers.


template<int Dt>
struct simd_upsampler 
{
    template<int P> 
    inline __m256 get() const;
};


template<> struct simd_upsampler<1> {
    __m256 t;
    simd_upsampler(__m256 t_) { t = t_; }

    template<int P> 
    inline __m256 get() const;
};

template<> inline __m256 simd_upsampler<1>::get<0> () const { return t; }


template<> struct simd_upsampler<2> {
    __m256 a, b;

    simd_upsampler(__m256 t)
    {
	__m256 u = _mm256_permute_ps(t, 0x50);   // [ t0 t0 t1 t1 t4 t4 t5 t5 ]
	__m256 v = _mm256_permute_ps(t, 0xfa);   // [ t2 t2 t3 t3 t6 t6 t7 t7 ]

	a = _mm256_permute2f128_ps(u, v, 0x20);  // [ t0 t0 t1 t1 t2 t2 t3 t3 ]
	b = _mm256_permute2f128_ps(u, v, 0x31);  // [ t4 t4 t5 t5 t6 t6 t7 t7 ]
    }

    template<int P> 
    inline __m256 get() const;
};

template<> inline __m256 simd_upsampler<2>::get<0> () const { return a; }
template<> inline __m256 simd_upsampler<2>::get<1> () const { return b; }


template<> struct simd_upsampler<4> {
    __m256 w;

    simd_upsampler(__m256 t) 
    {
	__m256 u = _mm256_permute_ps(t, 0xb1);          // [ t1 t0 t3 t2 t5 t4 t7 t6 ],  0xb1 = (2301)_4
	__m256 v = _mm256_permute2f128_ps(t, t, 0x01);  // [ t4 t5 t6 t7 t0 t1 t2 t3 ]
	w = _mm256_blend_ps(u, v, 0xa5);                // [ t4 t0 t6 t2 t5 t1 t7 t3 ],  0xa5 = (10100101)_2
    }

    template<int P> 
    inline __m256 get() const;
};

template<> inline __m256 simd_upsampler<4>::get<0> () const { return _mm256_permute_ps(w, 0x55); }  // [ t0 t0 t0 t0 t1 t1 t1 t1 ],  (1111)_4
template<> inline __m256 simd_upsampler<4>::get<1> () const { return _mm256_permute_ps(w, 0xff); }  // [ t2 t2 t2 t2 t3 t3 t3 t3 ],  (3333)_4
template<> inline __m256 simd_upsampler<4>::get<2> () const { return _mm256_permute_ps(w, 0x00); }  // [ t4 t4 t4 t4 t5 t5 t5 t5 ],  (0000)_4
template<> inline __m256 simd_upsampler<4>::get<3> () const { return _mm256_permute_ps(w, 0xaa); }  // [ t6 t6 t6 t6 t7 t7 t7 t7 ],  (2222)_4


template<> struct simd_upsampler<8> {
    __m256 u, v;

    simd_upsampler(__m256 t) 
    {
	__m256 r = _mm256_permute2f128_ps(t, t, 0x01);  // [t1 t0]

	u = _mm256_blend_ps(t, r, 0xf0);  // [t0 t0]
	v = _mm256_blend_ps(t, r, 0x0f);  // [t1 t1]
    }

    template<int P> 
    inline __m256 get() const;
};

template<> inline __m256 simd_upsampler<8>::get<0> () const { return _mm256_permute_ps(u, 0x00); }  // (0000)_4
template<> inline __m256 simd_upsampler<8>::get<1> () const { return _mm256_permute_ps(u, 0x55); }  // (1111)_4
template<> inline __m256 simd_upsampler<8>::get<2> () const { return _mm256_permute_ps(u, 0xaa); }  // (2222)_4
template<> inline __m256 simd_upsampler<8>::get<3> () const { return _mm256_permute_ps(u, 0xff); }  // (3333)_4
template<> inline __m256 simd_upsampler<8>::get<4> () const { return _mm256_permute_ps(v, 0x00); }
template<> inline __m256 simd_upsampler<8>::get<5> () const { return _mm256_permute_ps(v, 0x55); }
template<> inline __m256 simd_upsampler<8>::get<6> () const { return _mm256_permute_ps(v, 0xaa); }
template<> inline __m256 simd_upsampler<8>::get<7> () const { return _mm256_permute_ps(v, 0xff); }


// -------------------------------------------------------------------------------------------------
//
// simd_upsample<Dt>()
//
// Eventually, this code can be generalized to a pair <T,S> and moved to simd_helpers.

template<int P, int Dt, typename std::enable_if<(P==0),int>::type = 0>
inline void _partial_upsample(simd_ntuple<float,8,P> &dst, const simd_upsampler<Dt> &src)
{ }

template<int P, int Dt, typename std::enable_if<((P>0) && (P<=Dt)),int>::type = 0>
inline void _partial_upsample(simd_ntuple<float,8,P> &dst, const simd_upsampler<Dt> &src)
{ 
    _partial_upsample(dst.v, src);
    dst.x = src.template get<P-1> ();
}

template<int Dt>
inline simd_ntuple<float,8,Dt> simd_upsample(const simd_t<float,8> &t)
{
    simd_upsampler<Dt> u(t.x);
    simd_ntuple<float,8,Dt> ret;

    _partial_upsample(ret, u);
    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// upsample_and_mask<Df,Dt>()


// upsample_and_mask1(): applies mask to shape-(Df,S) array.
template<int Df, typename std::enable_if<(Df==0),int>::type = 0>
inline void _upsample_and_mask1(float *wp, __m256 mask, int stride)
{ }

template<int Df, typename std::enable_if<(Df>0),int>::type = 0>
inline void _upsample_and_mask1(float *wp, __m256 mask, int stride)
{
    _upsample_and_mask1<Df-1> (wp, mask, stride);

    float *p = wp + (Df-1)*stride;
    _mm256_storeu_ps(p, _mm256_and_ps(mask, _mm256_loadu_ps(p)));
}


// upsample_and_mask2(): applies mask to shape-(Df,P*S) array.
template<int Df, int Dt, int P=Dt, typename std::enable_if<(P==0),int>::type = 0>
inline void _upsample_and_mask2(float *wp, const simd_upsampler<Dt> &mask, int stride)
{ }

template<int Df, int Dt, int P=Dt, typename std::enable_if<(P>0),int>::type = 0>
inline void _upsample_and_mask2(float *wp, const simd_upsampler<Dt> &mask, int stride)
{   
    constexpr int Q = (P-1)*8;
    _upsample_and_mask2<Df,Dt,P-1> (wp, mask, stride);
    _upsample_and_mask1<Df> (wp+Q, mask.template get<P-1>(), stride);
}


// upsample_and_mask(): applies mask to shape-(Df,Dt*S) array.
template<int Df, int Dt>
inline void upsample_and_mask(float *wp, __m256 mask, int stride)
{
    simd_upsampler<Dt> u(mask);
    _upsample_and_mask2<Df,Dt> (wp, u, stride);
}


// -------------------------------------------------------------------------------------------------
//
// Caller must check all arguments, including these checks:
//   (nt_in % 8) == 0
//   Df == Df_
//   Dt == Dt_
//
// The (Df_, Dt_) arguments are superfluous!


template<int Df, int Dt>
inline void kernel_update_weights_Df_Dt(int nfreq_in, int nt_in, float *dst, int dstride, const float *src, int sstride, float w_cutoff_, int Df_, int Dt_)
{
    __m256 w_cutoff = _mm256_set1_ps(w_cutoff_);

    for (int ifreq = 0; ifreq < nfreq_in; ifreq++) {
	float *dst2 = dst + ifreq * Df * dstride;
	const float *src2 = src + ifreq * sstride;

	for (int it = 0; it < nt_in; it += 8) {
	    __m256 w = _mm256_loadu_ps(src2+it);
	    __m256 mask = _mm256_cmp_ps(w, w_cutoff, _CMP_GT_OQ);
	    upsample_and_mask<Df,Dt> (dst2 + it*Dt, mask, dstride);
	}
    }
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_UPSAMPLE_INTERNALS_HPP
