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
	
	out_ninvx += 32;
	out_ninv += 32;

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
	
	_mm256_storeu_ps(out_ninv, w11);
	_mm256_storeu_ps(out_ninv + 8, w12);
	_mm256_storeu_ps(out_ninv + 16, w13);
	_mm256_storeu_ps(out_ninv + 24, w22);
	_mm256_storeu_ps(out_ninv + 32, w23);
	_mm256_storeu_ps(out_ninv + 40, w33);
	
	out_ninv += 48;
    }
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_SPLINE_DETRENDER_INTERNAL_HPP
