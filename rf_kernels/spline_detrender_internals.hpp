#ifndef _RF_KERNELS_SPLINE_DETRENDER_INTERNALS_HPP
#define _RF_KERNELS_SPLINE_DETRENDER_INTERNALS_HPP

#include "immintrin.h"

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

namespace rf_kernels {
#if 0
}  // emacs pacifier
#endif


inline void spline_detrender::_kernel_ninv(int stride, const float *intensity, const float *weights)
{
    float *out_ninv = ninv;
    float *out_ninvx = ninvx;

    for (int b = 0; b < nbins; b++) {
	int ifreq0 = bin_delim[b];
	int ifreq1 = bin_delim[b+1];
	
	__m256 pp, p;

	// First pass starts here.
	
	__m256 wi0 = _mm256_setzero_ps();
	__m256 wi1 = _mm256_setzero_ps();
	__m256 wi2 = _mm256_setzero_ps();
	__m256 wi3 = _mm256_setzero_ps();
	__m256 w00 = _mm256_setzero_ps();
	__m256 w01 = _mm256_setzero_ps();
	__m256 w02 = _mm256_setzero_ps();
	__m256 w03 = _mm256_setzero_ps();
	
	pp = _mm256_load_ps(poly_vals + 4 * (ifreq0 & ~1));
	
	for (int ifreq = ifreq0; ifreq < ifreq1; ifreq++) {
	    __m256 w = _mm256_loadu_ps(weights + ifreq*stride);
	    __m256 wi = _mm256_loadu_ps(intensity + ifreq*stride);
	    wi *= w;
	    
	    // This branch inside the loop can be removed, by hand-unrolling the loop
	    // by a factor 2.  Strangely, this made performance worse! (gcc 4.8.5)
	    
	    if (ifreq & 1)
		p = _mm256_permute2f128_ps(pp, pp, 0x11);
	    else {
		pp = _mm256_load_ps(poly_vals + 4*ifreq);
		p = _mm256_permute2f128_ps(pp, pp, 0x00);
	    }
	    
	    __m256 p0 = _mm256_permute_ps(p, 0x00);
	    __m256 p1 = _mm256_permute_ps(p, 0x55);
	    __m256 p2 = _mm256_permute_ps(p, 0xaa);
	    __m256 p3 = _mm256_permute_ps(p, 0xff);
	    
	    wi0 += wi * p0;
	    wi1 += wi * p1;
	    wi2 += wi * p2;
	    wi3 += wi * p3;
	    
	    w *= p0;
	    w00 += w * p0;
	    w01 += w * p1;
	    w02 += w * p2;
	    w03 += w * p3;
	}
	
	_mm256_storeu_ps(out_ninvx, wi0);
	_mm256_storeu_ps(out_ninvx + 8, wi1);
	_mm256_storeu_ps(out_ninvx + 16, wi2);
	_mm256_storeu_ps(out_ninvx + 24, wi3);
	
	_mm256_storeu_ps(out_ninv, w00);
	_mm256_storeu_ps(out_ninv + 8, w01);
	_mm256_storeu_ps(out_ninv + 16, w02);
	_mm256_storeu_ps(out_ninv + 24, w03);

	// Second pass starts here.
	
	__m256 w11 = _mm256_setzero_ps();
	__m256 w12 = _mm256_setzero_ps();
	__m256 w13 = _mm256_setzero_ps();
	__m256 w22 = _mm256_setzero_ps();
	__m256 w23 = _mm256_setzero_ps();
	__m256 w33 = _mm256_setzero_ps();
	
	pp = _mm256_load_ps(poly_vals + 4 * (ifreq0 & ~1));
	
	for (int ifreq = ifreq0; ifreq < ifreq1; ifreq++) {
	    __m256 w = _mm256_loadu_ps(weights + ifreq*stride);
	    
	    if (ifreq & 1)
		p = _mm256_permute2f128_ps(pp, pp, 0x11);
	    else {
		pp = _mm256_load_ps(poly_vals + 4*ifreq);
		p = _mm256_permute2f128_ps(pp, pp, 0x00);
	    }
	    
	    __m256 p1 = _mm256_permute_ps(p, 0x55);
	    __m256 p2 = _mm256_permute_ps(p, 0xaa);
	    __m256 p3 = _mm256_permute_ps(p, 0xff);
	    
	    w11 += w * p1 * p1;
	    w12 += w * p1 * p2;
	    w13 += w * p1 * p3;
	    w22 += w * p2 * p2;
	    w23 += w * p2 * p3;
	    w33 += w * p3 * p3;
	}
	
	_mm256_storeu_ps(out_ninv + 32, w11);
	_mm256_storeu_ps(out_ninv + 40, w12);
	_mm256_storeu_ps(out_ninv + 48, w13);
	_mm256_storeu_ps(out_ninv + 56, w22);
	_mm256_storeu_ps(out_ninv + 64, w23);
	_mm256_storeu_ps(out_ninv + 72, w33);
	
	out_ninvx += 32;
	out_ninv += 80;
    }
}


inline void spline_detrender::_kernel_fit_pass1()
{
    constexpr float r00 = 6./5.;
    constexpr float r01 = 1./10.;
    constexpr float r11 = 2./15.;
    constexpr float r31 = -1./30.;

    __m256 two = _mm256_set1_ps(2.0);
    __m256 wsum = _mm256_setzero_ps();

    // Indirect way of computing the sum of the weights.
    for (int b = 0; b < nbins; b++) {
	wsum += _mm256_loadu_ps(ninv + 80*b);
	wsum += _mm256_loadu_ps(ninv + 80*b + 16) * two;
	wsum += _mm256_loadu_ps(ninv + 80*b + 56);
    }
    
    __m256 w = wsum * _mm256_set1_ps(epsilon / float(nbins));
    __m256 mask = _mm256_cmp_ps(wsum, _mm256_setzero_ps(), _CMP_GT_OQ);

    __m256 ls00 = _mm256_setzero_ps();
    __m256 ls01 = _mm256_setzero_ps();
    __m256 ls10 = _mm256_setzero_ps();
    __m256 ls11 = _mm256_setzero_ps();

    int b = 0;

    for (;;) {

	// Compute cholesky_invdiag[b].
	// Here, 0 <= b <= nbins.

	__m256 ninv00 = _mm256_setzero_ps();
	__m256 ninv01 = _mm256_setzero_ps();
	__m256 ninv11 = _mm256_setzero_ps();

	if (b > 0) {
	    ninv00 += _mm256_loadu_ps(ninv + 80*(b-1) + 56);  // w22
	    ninv01 += _mm256_loadu_ps(ninv + 80*(b-1) + 64);  // w23
	    ninv11 += _mm256_loadu_ps(ninv + 80*(b-1) + 72);  // w33

	    ninv00 += w * _mm256_set1_ps(r00);
	    ninv01 -= w * _mm256_set1_ps(r01);   // note minus sign here
	    ninv11 += w * _mm256_set1_ps(r11);

	    ninv00 -= (ls00*ls00 + ls01*ls01);
	    ninv01 -= (ls00*ls10 + ls01*ls11);
	    ninv11 -= (ls10*ls10 + ls11*ls11);
	}

	if (b < nbins) {
	    ninv00 += _mm256_loadu_ps(ninv + 80*b);       // w00
	    ninv01 += _mm256_loadu_ps(ninv + 80*b + 8);   // w01
	    ninv11 += _mm256_loadu_ps(ninv + 80*b + 32);  // w11

	    ninv00 += w * _mm256_set1_ps(r00);
	    ninv01 += w * _mm256_set1_ps(r01);
	    ninv11 += w * _mm256_set1_ps(r11);
	}

	// Cholesky-factorize 2-by-2 matrix.
	// I decided to use _mm256_sqrt_ps(), rather than the faster-but-less-accurate _mm256_rsqrt_ps().

	__m256 zero = _mm256_setzero_ps();
	__m256 one = _mm256_set1_ps(1.0);
	__m256 l00 = _mm256_blendv_ps(one, _mm256_sqrt_ps(ninv00), mask);
	__m256 linv00 = _mm256_blendv_ps(zero, one/l00, mask);
	__m256 l10 = ninv01 * linv00;
	__m256 l11 = _mm256_blendv_ps(one, _mm256_sqrt_ps(ninv11-l10*l10), mask);
	__m256 linv11 = _mm256_blendv_ps(zero, one/l11, mask);
	__m256 linv10 = -l10 * linv00 * linv11;
	
	_mm256_storeu_ps(cholesky_invdiag + 24*b, linv00);
	_mm256_storeu_ps(cholesky_invdiag + 24*b + 8, linv10);
	_mm256_storeu_ps(cholesky_invdiag + 24*b + 16, linv11);

	if (b == nbins)
	    return;

	// Compute cholesky_subdiag[b].
	// Here, 0 <= b < nbins.

	__m256 s00 = _mm256_loadu_ps(ninv + 80*b + 16);  // w20
	__m256 s01 = _mm256_loadu_ps(ninv + 80*b + 40);  // w21
	__m256 s10 = _mm256_loadu_ps(ninv + 80*b + 24);  // w30
	__m256 s11 = _mm256_loadu_ps(ninv + 80*b + 48);  // w31

	s00 -= w * _mm256_set1_ps(r00);  // r20 = -r00
	s01 -= w * _mm256_set1_ps(r01);  // r21 = -r01
	s10 += w * _mm256_set1_ps(r01);  // r30 = r01
	s11 += w * _mm256_set1_ps(r31);

	// Multiply on the right by L^{-T}
	ls00 = s00 * linv00;
	ls01 = s00 * linv10 + s01 * linv11;
	ls10 = s10 * linv00;
	ls11 = s10 * linv10 + s11 * linv11;

	_mm256_storeu_ps(cholesky_subdiag + 32*b, ls00);
	_mm256_storeu_ps(cholesky_subdiag + 32*b + 8, ls01);
	_mm256_storeu_ps(cholesky_subdiag + 32*b + 16, ls10);
	_mm256_storeu_ps(cholesky_subdiag + 32*b + 24, ls11);

	b++;
    }
}


inline void spline_detrender::_kernel_fit_pass2()
{
    __m256 w0 = _mm256_setzero_ps();
    __m256 w1 = _mm256_setzero_ps();

    for (int b = 0; b <= nbins; b++) {
	__m256 v0 = _mm256_setzero_ps();
	__m256 v1 = _mm256_setzero_ps();

	if (b > 0) {
	    __m256 s00 = _mm256_loadu_ps(cholesky_subdiag + 32*(b-1));
	    __m256 s01 = _mm256_loadu_ps(cholesky_subdiag + 32*(b-1) + 8);
	    __m256 s10 = _mm256_loadu_ps(cholesky_subdiag + 32*(b-1) + 16);
	    __m256 s11 = _mm256_loadu_ps(cholesky_subdiag + 32*(b-1) + 24);

	    v0 += _mm256_loadu_ps(ninvx + 32*(b-1) + 16);
	    v1 += _mm256_loadu_ps(ninvx + 32*(b-1) + 24);

	    v0 -= (s00*w0 + s01*w1);
	    v1 -= (s10*w0 + s11*w1);
	}

	if (b < nbins) {
	    v0 += _mm256_loadu_ps(ninvx + 32*b);
	    v1 += _mm256_loadu_ps(ninvx + 32*b + 8);
	}

	__m256 linv00 = _mm256_loadu_ps(cholesky_invdiag + 24*b);
	__m256 linv10 = _mm256_loadu_ps(cholesky_invdiag + 24*b + 8);
	__m256 linv11 = _mm256_loadu_ps(cholesky_invdiag + 24*b + 16);

	w0 = linv00*v0;
	w1 = linv10*v0 + linv11*v1;

	_mm256_storeu_ps(coeffs + 16*b, w0);
	_mm256_storeu_ps(coeffs + 16*b + 8, w1);
    }
}


inline void spline_detrender::_kernel_fit_pass3()
{
    __m256 w0 = _mm256_setzero_ps();
    __m256 w1 = _mm256_setzero_ps();

    for (int b = nbins; b >= 0; b--) {
	__m256 v0 = _mm256_loadu_ps(coeffs + 16*b);
	__m256 v1 = _mm256_loadu_ps(coeffs + 16*b + 8);
	
	if (b < nbins) {
	    __m256 st00 = _mm256_loadu_ps(cholesky_subdiag + 32*b);
	    __m256 st10 = _mm256_loadu_ps(cholesky_subdiag + 32*b + 8);
	    __m256 st01 = _mm256_loadu_ps(cholesky_subdiag + 32*b + 16);
	    __m256 st11 = _mm256_loadu_ps(cholesky_subdiag + 32*b + 24);
	    
	    v0 -= (st00*w0 + st01*w1);
	    v1 -= (st10*w0 + st11*w1);
	} 
	 
	__m256 ltinv00 = _mm256_loadu_ps(cholesky_invdiag + 24*b);
	__m256 ltinv01 = _mm256_loadu_ps(cholesky_invdiag + 24*b + 8);
	__m256 ltinv11 = _mm256_loadu_ps(cholesky_invdiag + 24*b + 16);

	w0 = ltinv00*v0 + ltinv01*v1;
	w1 = ltinv11*v1;

	_mm256_storeu_ps(coeffs + 16*b, w0);
	_mm256_storeu_ps(coeffs + 16*b + 8, w1);
    }
}


inline void spline_detrender::_kernel_detrend(int stride, float *intensity)
{
    __m256 c0, c1;
    __m256 c2 = _mm256_loadu_ps(coeffs);
    __m256 c3 = _mm256_loadu_ps(coeffs + 8);

    for (int b = 0; b < nbins; b++) {
	int ifreq0 = bin_delim[b];
	int ifreq1 = bin_delim[b+1];

	c0 = c2;
	c1 = c3;
	c2 = _mm256_loadu_ps(coeffs + 16*b + 16);
	c3 = _mm256_loadu_ps(coeffs + 16*b + 24);

	__m256 pp = _mm256_load_ps(poly_vals + 4 * (ifreq0 & ~1));
	__m256 p;
	
	for (int ifreq = ifreq0; ifreq < ifreq1; ifreq++) {
	    __m256 ival = _mm256_loadu_ps(intensity + ifreq*stride);

	    if (ifreq & 1)
		p = _mm256_permute2f128_ps(pp, pp, 0x11);
	    else {
		pp = _mm256_load_ps(poly_vals + 4*ifreq);
		p = _mm256_permute2f128_ps(pp, pp, 0x00);
	    }
	    
	    __m256 p0 = _mm256_permute_ps(p, 0x00);
	    __m256 p1 = _mm256_permute_ps(p, 0x55);
	    __m256 p2 = _mm256_permute_ps(p, 0xaa);
	    __m256 p3 = _mm256_permute_ps(p, 0xff);

	    ival -= (c0*p0 + c1*p1 + c2*p2 + c3*p3);

	    _mm256_storeu_ps(intensity + ifreq*stride, ival);
	}
    }	
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_SPLINE_DETRENDER_INTERNALS_HPP
