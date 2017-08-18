#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"

#include "simd_helpers/simd_float32.hpp"

using namespace std;
using namespace rf_kernels;
using namespace simd_helpers;


static void _sum_weights(int nfreq, int nt_chunk, int stride, float *intensity, float *weights, float *poly_vals, float *out)
{
    for (int it = 0; it < nt_chunk; it += 8) {
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
	    simd_t<float,8> p = simd_load<float,8> (poly_vals + 8*ifreq);
	    simd_t<float,8> p0 = _mm256_permute_ps(p.x, 0x00);
	    simd_t<float,8> p1 = _mm256_permute_ps(p.x, 0x55);
	    simd_t<float,8> p2 = _mm256_permute_ps(p.x, 0xaa);
	    simd_t<float,8> p3 = _mm256_permute_ps(p.x, 0xff);

	    simd_t<float,8> w = simd_load<float,8> (weights + ifreq*stride + it);

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
	}

	simd_store(out + 10*it, w00);
	simd_store(out + 10*it + 8, w01);
	simd_store(out + 10*it + 16, w02);
	simd_store(out + 10*it + 24, w03);
	simd_store(out + 10*it + 32, w11);
	simd_store(out + 10*it + 40, w12);
	simd_store(out + 10*it + 48, w13);
	simd_store(out + 10*it + 56, w22);
	simd_store(out + 10*it + 64, w23);
	simd_store(out + 10*it + 72, w33);
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
	const int nchunks = 256;

	this->start_timer();

	for (int i = 0; i < nchunks; i++)
	    _sum_weights(nfreq, nt_chunk, stride, intensity, weights, poly_vals, tmp);
	
	double dt = this->stop_timer();
	ssize_t nsamples = ssize_t(nchunks) * ssize_t(nfreq) * ssize_t(nt_chunk);

	if (thread_id == 0) {
	    cout << "spline_detrender: " << nchunks << " chunks,"
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