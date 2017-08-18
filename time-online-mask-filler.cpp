#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/online_mask_filler.hpp"

using namespace std;
using namespace rf_kernels;


struct online_mask_filler_timing_thread : public kernel_timing_thread {
    online_mask_filler_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override 
    {
	online_mask_filler_params kparams;

	float *running_variance = aligned_alloc<float> (nfreq);
	float *running_weights = aligned_alloc<float> (nfreq);
	uint64_t rng_state[8];

	this->allocate();
	this->start_timer();

	for (int i = 0; i < niter; i++)
	    online_mask_fill(kparams, nfreq, nt_chunk, stride, intensity, weights, running_variance, running_weights, rng_state);

	this->stop_timer2("online_mask_filler");
    
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
