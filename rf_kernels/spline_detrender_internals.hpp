#ifndef _RF_KERNELS_SPLINE_DETRENDER_INTERNAL_HPP
#define _RF_KERNELS_SPLINE_DETRENDER_INTERNAL_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++0x support (g++ -std=c++0x)"
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
    __m256 two = _mm256_set1_ps(2.0);
    __m256 wsum = _mm256_setzero_ps();

    // Indirect way of computing the sum of the weights.
    for (int b = 0; b < nbins; b++) {
	wsum += _mm256_loadu_ps(ninv + 80*b);
	wsum += _mm256_loadu_ps(ninv + 80*b + 16) * two;
	wsum += _mm256_loadu_ps(ninv + 80*b + 56);
    }
    
    __m256 w = wsum * _mm256_set1_ps(epsilon / float(nbins));
    __m256 mask = _mm256_cmp_ps(w, _mm256_setzero_ps(), _CMP_GT_OQ);

    constexpr float r00 = 6./5.;
    constexpr float r01 = 1./10.;
    constexpr float r11 = 2./15.;

    int b = 0;

    for (;;) {

	// Compute cholesky_invdiag[b].
	// Here, 0 <= b <= nbins.

	__m256 ninv00 = w * _mm256_set1_ps(r00);
	__m256 ninv01 = w * _mm256_set1_ps(r01);
	__m256 ninv11 = w * _mm256_set1_ps(r11);

	if (b > 0) {
	    ninv00 += _mm256_loadu_ps(ninv + 80*(b-1) + 56);  // w22
	    ninv01 += _mm256_loadu_ps(ninv + 80*(b-1) + 64);  // w23
	    ninv11 += _mm256_loadu_ps(ninv + 80*(b-1) + 72);  // w33
	}

	if (b < nbins) {
	    ninv00 += _mm256_loadu_ps(ninv + 80*b);       // w00
	    ninv01 += _mm256_loadu_ps(ninv + 80*b + 8);   // w01
	    ninv11 += _mm256_loadu_ps(ninv + 80*b + 32);  // w11
	}

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

#if 0
	// Compute cholesky_subdiag[b].
	// Here, 0 <= b < nbins.

	ninv00 = w * _mm256_set1_ps();
	ninv01 = w * _mm256_set1_ps();
	ninv10 = w * _mm256_set1_ps();
	ninv11 = w * _mm256_set1_ps();

	if (b > 0) {
	}

	if (b < nbins) {
	}

	_mm256_storeu_ps(cholesky_subdiag + , );
#endif
    }
}



inline void spline_detrender::_kernel_detrend(int stride, float *intensity)
{
    for (int b = 0; b < nbins; b++) {
	int ifreq0 = bin_delim[b];
	int ifreq1 = bin_delim[b+1];

	__m256 c0 = _mm256_loadu_ps(coeffs + 32*b);
	__m256 c1 = _mm256_loadu_ps(coeffs + 32*b + 8);
	__m256 c2 = _mm256_loadu_ps(coeffs + 32*b + 16);
	__m256 c3 = _mm256_loadu_ps(coeffs + 32*b + 24);
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

#endif  // _RF_KERNELS_SPLINE_DETRENDER_INTERNAL_HPP
