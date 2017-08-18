#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"

#include "simd_helpers/simd_float32.hpp"

using namespace std;
using namespace rf_kernels;
using namespace simd_helpers;


static void _one_pass(int nfreq, int nt_chunk, int stride, float *intensity, float *weights, float *poly_vals, float *out)
{
    for (int it = 0; it < nt_chunk; it += 8) {
	simd_t<float,8> wi0(0.0);
	simd_t<float,8> wi1(0.0);
	simd_t<float,8> wi2(0.0);
	simd_t<float,8> wi3(0.0);

	simd_t<float,8> w00(0.0);
	simd_t<float,8> w01(0.0);
	simd_t<float,8> w02(0.0);
	simd_t<float,8> w03(0.0);
	simd_t<float,8> w11(0.0);
	simd_t<float,8> w12(0.0);
	simd_t<float,8> w13(0.0);
	simd_t<float,8> w22(0.0);
	simd_t<float,8> w23(0.0);
	simd_t<float,8> w33(0.0);

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_t<float,8> w = simd_load<float,8> (weights + ifreq*stride + it);
	    simd_t<float,8> wi = simd_load<float,8> (intensity + ifreq*stride + it);
	    wi *= w;

	    simd_t<float,8> p = simd_load<float,8> (poly_vals + 8*ifreq);
	    simd_t<float,8> p0 = _mm256_permute_ps(p.x, 0x00);
	    simd_t<float,8> p1 = _mm256_permute_ps(p.x, 0x55);
	    simd_t<float,8> p2 = _mm256_permute_ps(p.x, 0xaa);
	    simd_t<float,8> p3 = _mm256_permute_ps(p.x, 0xff);

	    wi0 += wi * p0;
	    wi1 += wi * p1;
	    wi2 += wi * p2;
	    wi3 += wi * p3;

#if 1
	    // Seems to be faster 
	    w00 += w * p0 * p0;
	    w01 += w * p0 * p1;
	    w02 += w * p0 * p2;
	    w03 += w * p0 * p3;
	    w11 += w * p1 * p1;
	    w12 += w * p1 * p2;
	    w13 += w * p1 * p3;
	    w22 += w * p2 * p2;
	    w23 += w * p2 * p3;
	    w33 += w * p3 * p3;
#else
	    simd_t<float,8> q0 = p0 * p.x;
	    simd_t<float,8> q1 = p1 * p.x;
	    simd_t<float,8> q2 = _mm256_permute_ps(p.x, 0xfa) * _mm256_permute_ps(p.x, 0xee);

	    w00 += w * _mm256_permute_ps(q0.x, 0x00);
	    w01 += w * _mm256_permute_ps(q0.x, 0x55);
	    w02 += w * _mm256_permute_ps(q0.x, 0xaa);
	    w03 += w * _mm256_permute_ps(q0.x, 0xff);
	    w11 += w * _mm256_permute_ps(q1.x, 0x55);
	    w12 += w * _mm256_permute_ps(q1.x, 0xaa);
	    w13 += w * _mm256_permute_ps(q1.x, 0xff);
	    w22 += w * _mm256_permute_ps(q2.x, 0x00);
	    w23 += w * _mm256_permute_ps(q2.x, 0x55);
	    w33 += w * _mm256_permute_ps(q2.x, 0xff);
#endif
	}

	simd_store(out + 14*it, wi0);
	simd_store(out + 14*it + 8, wi1);
	simd_store(out + 14*it + 16, wi2);
	simd_store(out + 14*it + 24, wi3);
	simd_store(out + 14*it + 32, w00);
	simd_store(out + 14*it + 40, w01);
	simd_store(out + 14*it + 48, w02);
	simd_store(out + 14*it + 56, w03);
	simd_store(out + 14*it + 64, w11);
	simd_store(out + 14*it + 72, w12);
	simd_store(out + 14*it + 80, w13);
	simd_store(out + 14*it + 88, w22);
	simd_store(out + 14*it + 96, w23);
	simd_store(out + 14*it + 104, w33);
    }
}


static void _two_pass(int nfreq, int nt_chunk, int stride, float *intensity, float *weights, float *poly_vals, float *out)
{
    for (int it = 0; it < nt_chunk; it += 8) {
	simd_t<float,8> wi0(0.0);
	simd_t<float,8> wi1(0.0);
	simd_t<float,8> wi2(0.0);
	simd_t<float,8> wi3(0.0);
	simd_t<float,8> w00(0.0);
	simd_t<float,8> w01(0.0);
	simd_t<float,8> w02(0.0);
	simd_t<float,8> w03(0.0);

	// First pass
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_t<float,8> w = simd_load<float,8> (weights + ifreq*stride + it);
	    simd_t<float,8> wi = simd_load<float,8> (intensity + ifreq*stride + it);
	    wi *= w;

	    simd_t<float,8> p = simd_load<float,8> (poly_vals + 8*ifreq);
	    simd_t<float,8> p0 = _mm256_permute_ps(p.x, 0x00);
	    simd_t<float,8> p1 = _mm256_permute_ps(p.x, 0x55);
	    simd_t<float,8> p2 = _mm256_permute_ps(p.x, 0xaa);
	    simd_t<float,8> p3 = _mm256_permute_ps(p.x, 0xff);

	    wi0 += wi * p0;
	    wi1 += wi * p1;
	    wi2 += wi * p2;
	    wi3 += wi * p3;

	    w00 += w * p0 * p0;
	    w01 += w * p0 * p1;
	    w02 += w * p0 * p2;
	    w03 += w * p0 * p3;
	}

	simd_store(out + 14*it, wi0);
	simd_store(out + 14*it + 8, wi1);
	simd_store(out + 14*it + 16, wi2);
	simd_store(out + 14*it + 24, wi3);
	simd_store(out + 14*it + 32, w00);
	simd_store(out + 14*it + 40, w01);
	simd_store(out + 14*it + 48, w02);
	simd_store(out + 14*it + 56, w03);

	simd_t<float,8> w11(0.0);
	simd_t<float,8> w12(0.0);
	simd_t<float,8> w13(0.0);
	simd_t<float,8> w22(0.0);
	simd_t<float,8> w23(0.0);
	simd_t<float,8> w33(0.0);

	// Second pass
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_t<float,8> w = simd_load<float,8> (weights + ifreq*stride + it);

	    simd_t<float,8> p = simd_load<float,8> (poly_vals + 8*ifreq);
	    simd_t<float,8> p1 = _mm256_permute_ps(p.x, 0x55);
	    simd_t<float,8> p2 = _mm256_permute_ps(p.x, 0xaa);
	    simd_t<float,8> p3 = _mm256_permute_ps(p.x, 0xff);

	    w11 += w * p1 * p1;
	    w12 += w * p1 * p2;
	    w13 += w * p1 * p3;
	    w22 += w * p2 * p2;
	    w23 += w * p2 * p3;
	    w33 += w * p3 * p3;
	}

	simd_store(out + 14*it + 64, w11);
	simd_store(out + 14*it + 72, w12);
	simd_store(out + 14*it + 80, w13);
	simd_store(out + 14*it + 88, w22);
	simd_store(out + 14*it + 96, w23);
	simd_store(out + 14*it + 104, w33);
    }
}


struct spline_detrender_timing_thread : public timing_thread {
    const kernel_timing_params params;

    spline_detrender_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	timing_thread(pool_, true),   // pin_to_core=true
	params(params_)
    { }

    virtual ~spline_detrender_timing_thread() { }

    virtual void thread_body() override 
    {
	int nfreq = params.nfreq;
	int stride = params.stride;
	int nt_chunk = params.nt_chunk;

	// Put memory allocations in thread_body(), not constructor, so that
	// memory is allocated by the same CPU that runs the kernel.
	//
	// Note: no need to initialize intensity/weights arrays, since
	// spline_detrender kernel contains no branches, so the timing
	// will be independent of the array contents.
	
	float *intensity = aligned_alloc<float> (nfreq * stride);
	float *weights = aligned_alloc<float> (nfreq * stride);
	float *poly_vals = aligned_alloc<float> (64 * nfreq);
	float *tmp = aligned_alloc<float> (64 * nt_chunk);

	// Chosen arbitrarily!
	const int nchunks = 64;

	this->start_timer();

	for (int i = 0; i < nchunks; i++)
	    _one_pass(nfreq, nt_chunk, stride, intensity, weights, poly_vals, tmp);
	
	double dt = this->stop_timer();
	ssize_t nsamples = ssize_t(nchunks) * ssize_t(nfreq) * ssize_t(nt_chunk);

	if (thread_id == 0) {
	    cout << "one_pass: " << nchunks << " chunks,"
		 << " total time " << dt << " sec"
		 << " (" << (dt/nchunks) << " sec/chunk,"
		 << " " << (1.0e9 * dt / nsamples) << " ns/sample)" << endl;
	}

	this->start_timer();

	for (int i = 0; i < nchunks; i++)
	    _two_pass(nfreq, nt_chunk, stride, intensity, weights, poly_vals, tmp);
	
	dt = this->stop_timer();
	nsamples = ssize_t(nchunks) * ssize_t(nfreq) * ssize_t(nt_chunk);

	if (thread_id == 0) {
	    cout << "two_pass: " << nchunks << " chunks,"
		 << " total time " << dt << " sec"
		 << " (" << (dt/nchunks) << " sec/chunk,"
		 << " " << (1.0e9 * dt / nsamples) << " ns/sample)" << endl;
	}

	free(intensity);
	free(weights);
	free(tmp);
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
