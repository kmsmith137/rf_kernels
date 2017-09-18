#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/polynomial_detrender.hpp"
// #include "rf_kernels/polynomial_detrender_internals.hpp"


using namespace std;
using namespace rf_kernels;


struct polynomial_detrender_timing_thread : public kernel_timing_thread {
    polynomial_detrender_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override
    {
	this->allocate();

	for (int polydeg: { 1, 2, 3, 4, 6, 8 }) {
	    for (axis_type axis: { AXIS_FREQ, AXIS_TIME }) {
		stringstream ss;
		ss << "polynomial_detrender(axis=" << axis << ",polydeg=" << polydeg << ")";

		string s = ss.str();
		const char *cp = s.c_str();

		// I don't think this is necessary, but just being paranoid...
		for (int i = 0; i < nfreq*stride; i++)
		    weights[i] = 1.0;

		polynomial_detrender pd(axis, polydeg);

		this->start_timer();

		for (int iter = 0; iter < niter; iter++)
		    pd.detrend(nfreq, nt_chunk, intensity, stride, weights, stride, 1.0e-2);

		this->stop_timer2(cp);
	    }
	}
    }
};


int main(int argc, char **argv)
{
    kernel_timing_params params("time-polynomial-detrender");
    params.parse_args(argc, argv);

    int nthreads = params.nthreads;
    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<polynomial_detrender_timing_thread> (pool, params));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
