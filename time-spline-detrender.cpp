#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/spline_detrender.hpp"


using namespace std;
using namespace rf_kernels;


static void _two_pass(int nfreq, int nt_chunk, int stride, int nbins, const int *bin_delim, float *intensity, float *weights, const float *poly_vals, float *out)
{
    for (int it = 0; it < nt_chunk; it += 8) {
	
	for (int b = 0; b < nbins; b++) {
	    int ifreq0 = bin_delim[b];
	    int ifreq1 = bin_delim[b+1];

	    __m256 pp, p;

#if 1
	    // First pass.

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
		__m256 w = _mm256_loadu_ps(weights + ifreq*stride + it);
		__m256 wi = _mm256_loadu_ps(intensity + ifreq*stride + it);
		wi *= w;
		
		// This branch inside the loop can be removed, by hand-unrolling
		// the loop by a factor 2.  Strangely, this makes performance worse!
		
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
	    
	    _mm256_storeu_ps(out, wi0);
	    _mm256_storeu_ps(out + 8, wi1);
	    _mm256_storeu_ps(out + 16, wi2);
	    _mm256_storeu_ps(out + 24, wi3);
	    _mm256_storeu_ps(out + 32, w00);
	    _mm256_storeu_ps(out + 40, w01);
	    _mm256_storeu_ps(out + 48, w02);
	    _mm256_storeu_ps(out + 56, w03);
	    
	    out += 64;
#endif

#if 1
	    // Second pass

	    __m256 w11 = _mm256_setzero_ps();
	    __m256 w12 = _mm256_setzero_ps();
	    __m256 w13 = _mm256_setzero_ps();
	    __m256 w22 = _mm256_setzero_ps();
	    __m256 w23 = _mm256_setzero_ps();
	    __m256 w33 = _mm256_setzero_ps();

	    pp = _mm256_load_ps(poly_vals + 4 * (ifreq0 & ~1));

	    for (int ifreq = ifreq0; ifreq < ifreq1; ifreq++) {
		__m256 w = _mm256_loadu_ps(weights + ifreq*stride + it);
		
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
	    
	    _mm256_storeu_ps(out, w11);
	    _mm256_storeu_ps(out + 8, w12);
	    _mm256_storeu_ps(out + 16, w13);
	    _mm256_storeu_ps(out + 24, w22);
	    _mm256_storeu_ps(out + 32, w23);
	    _mm256_storeu_ps(out + 40, w33);

	    out += 48;
#endif
	}
    }
}


struct spline_detrender_timing_thread : public kernel_timing_thread {
    spline_detrender_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override 
    {
	const int nbins = 4;
	vector<int> bin_delim(nbins+1, 0);

	float *poly_vals = aligned_alloc<float> (4 * nfreq);
	float *out = aligned_alloc<float> (14 * nbins * nt_chunk);

	_spline_detrender_init(&bin_delim[0], poly_vals, nfreq, nbins);

	this->allocate();
	this->start_timer();

	for (int i = 0; i < niter; i++)
	    _two_pass(nfreq, nt_chunk, stride, nbins, &bin_delim[0], intensity, weights, poly_vals, out);

	this->stop_timer2("spline_detrender");

	free(poly_vals);
	free(out);
    }
};


int main(int argc, char **argv)
{
    kernel_timing_params params("time-spline-detrender");
    params.parse_args(argc, argv);

    int nthreads = params.nthreads;
    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<spline_detrender_timing_thread> (pool, params));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
