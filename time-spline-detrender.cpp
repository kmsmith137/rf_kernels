#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/spline_detrender.hpp"
#include "rf_kernels/spline_detrender_internals.hpp"


using namespace std;
using namespace rf_kernels;


struct spline_detrender_timing_thread : public kernel_timing_thread {
    spline_detrender_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override
    {
	const int nbins = 4;

	spline_detrender sd(nfreq, nbins);

	this->allocate();
	this->start_timer();

	for (int iter = 0; iter < niter; iter++)
	    sd.detrend(nt_chunk, stride, intensity, weights);

	this->stop_timer2("spline_detrender");
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
