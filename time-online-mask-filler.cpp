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
	// bonsai-like parameters
	online_mask_filler mf(nfreq);
	mf.v1_chunk = 32;
	mf.var_weight = 0.01;
	mf.w_clamp = 0.01;
	mf.w_cutoff = 0.1;
	mf.modify_weights = false;
	mf.multiply_intensity_by_weights = true;

	this->allocate();
	this->start_timer();

	for (int i = 0; i < niter; i++)
	    mf.mask_fill(nt_chunk, stride, intensity, weights);

	this->stop_timer2("online_mask_filler");
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
