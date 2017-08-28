#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/upsample.hpp"
#include "rf_kernels/upsample_internals.hpp"


using namespace std;
using namespace rf_kernels;


struct upsample_timing_thread : public kernel_timing_thread {
    upsample_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override
    {
	this->allocate();

	for (int Df: { 1, 2, 4, 8, 16 }) {
	    for (int Dt: { 1, 2, 4, 8, 16 }) {
		stringstream ss;
		ss << "weight_upsampler(Df=" << Df << ",Dt=" << Dt << ")";

		string s = ss.str();
		const char *cp = s.c_str();

		weight_upsampler u(Df,Dt);
		int nfreq_in = xdiv(nfreq, Df);
		int nt_in = xdiv(nt_chunk, Dt);

		this->start_timer();

		for (int iter = 0; iter < niter; iter++) {
		    // Note: we use this->intensity as the output buffer (hires weights),
		    // and this->weights as the input buffer (lores weights).  We use the
		    // same stride for both arrays, even though the lores weights will
		    // probably have a smaller stride in practice.

		    u.upsample(nfreq_in, nt_in, intensity, stride, weights, stride);
		}

		this->stop_timer2(cp);
	    }
	}
    }
};


int main(int argc, char **argv)
{
    kernel_timing_params params("time-upsample");
    params.parse_args(argc, argv);

    int nthreads = params.nthreads;
    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<upsample_timing_thread> (pool, params));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
