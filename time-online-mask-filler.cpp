#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/online_mask_filler.hpp"

using namespace std;
using namespace rf_kernels;


struct online_mask_filler_timing_thread : public timing_thread {
    const kernel_timing_params params;

    online_mask_filler_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	timing_thread(pool_, true),   // pin_to_core=true
	params(params_)
    { }

    virtual ~online_mask_filler_timing_thread() { }

    virtual void thread_body() override 
    {
	online_mask_filler_params kparams;

	int nfreq = params.nfreq;
	int stride = params.stride;
	int nt_chunk = params.nt_chunk;

	// Put memory allocations in thread_body(), not constructor, so that
	// memory is allocated by the same CPU that runs the kernel.
	//
	// Note: no need to initialize intensity/weights arrays, since
	// online_mask_filler kernel contains no branches, so the timing
	// will be independent of the array contents.
	
	float *intensity = aligned_alloc<float> (nfreq * stride);
	float *weights = aligned_alloc<float> (nfreq * stride);
	float *running_variance = aligned_alloc<float> (nfreq);
	float *running_weights = aligned_alloc<float> (nfreq);
	uint64_t rng_state[8];

	// Chosen arbitrarily!
	const int nchunks = 64;

	this->start_timer();

	for (int i = 0; i < nchunks; i++)
	    online_mask_fill(kparams, nfreq, nt_chunk, stride, intensity, weights, running_variance, running_weights, rng_state);
	
	double dt = this->stop_timer();
	ssize_t nsamples = ssize_t(nchunks) * ssize_t(nfreq) * ssize_t(nt_chunk);

	if (thread_id == 0) {
	    cout << "online_mask_filler: " << nchunks << " chunks,"
		 << " total time " << dt << " sec"
		 << " (" << (dt/nchunks) << " sec/chunk,"
		 << " " << (1.0e9 * dt / nsamples) << " ns/sample)" << endl;
	}

	free(intensity);
	free(weights);
	free(running_variance);
	free(running_weights);
    }
};


int main(int argc, char **argv)
{
    kernel_timing_params params("time-online-mask-filler");
    params.parse_args(argc, argv);

    int nthreads = params.nthreads;
    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<online_mask_filler_timing_thread> (pool, params));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
