#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/downsample.hpp"


using namespace std;
using namespace rf_kernels;


struct downsample_timing_thread : public kernel_timing_thread {
    downsample_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override
    {
	this->allocate();

	for (int Df: { 1, 2, 4, 8, 16, 64, 256 }) {
	    for (int Dt: { 1, 2, 4, 8, 16 }) {
		stringstream ss;
		ss << "wi_downsampler(Df=" << Df << ",Dt=" << Dt << ")";

		string s = ss.str();
		const char *cp = s.c_str();

		wi_downsampler ds(Df,Dt);
		int nfreq_ds = xdiv(nfreq, Df);
		int nt_ds = xdiv(nt_chunk, Dt);

		// Use same stride for input and output arrays (a little unrealistic)
		float *intensity_ds = aligned_alloc<float> (nfreq_ds * stride);
		float *weights_ds = aligned_alloc<float> (nfreq_ds * stride);

		this->start_timer();

		for (int iter = 0; iter < niter; iter++)
		    ds.downsample(nfreq_ds, nt_ds, intensity_ds, stride, weights_ds, stride, intensity, stride, weights, stride);

		this->stop_timer2(cp);

		free(intensity_ds);
		free(weights_ds);
	    }
	}
    }
};


int main(int argc, char **argv)
{
    kernel_timing_params params("time-downsample");
    params.parse_args(argc, argv);

    int nthreads = params.nthreads;
    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<downsample_timing_thread> (pool, params));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
