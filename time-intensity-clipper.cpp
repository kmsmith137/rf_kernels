#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/intensity_clipper.hpp"


using namespace std;
using namespace rf_kernels;


struct intensity_clipper_timing_thread : public kernel_timing_thread {
    intensity_clipper_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override
    {
	this->allocate();

	// Note: 'ic_niter' is the number of clipper iterations; this->niter is the number of "outer" timing iterations!
	for (int Df: { 1, 2, 4, 8, 16, 64, 256 }) {
	    for (int Dt: { 1, 2, 4, 8, 16 }) {
		// for (axis_type axis: { AXIS_FREQ, AXIS_TIME, AXIS_NONE })  {
		for (axis_type axis: { AXIS_FREQ, AXIS_TIME }) {
		    for (int ic_niter: {1,4}) {
			// for (bool two_pass: {false,true}) {
			for (bool two_pass: {true}) {
			    if ((ic_niter > 1) && !two_pass)
				continue;  // don't bother timing this case.
			    
			    stringstream ss;
			    ss << "intensity_clipper(axis=" << axis << ",Df=" << Df << ",Dt=" << Dt << ",niter=" << ic_niter << ",two_pass=" << two_pass << ")";
			    
			    string s = ss.str();
			    const char *cp = s.c_str();
			    
			    // Use a huge 'sigma', since we iterate the clipper many times in the loop below.
			    intensity_clipper ic(nfreq, nt_chunk, axis, 10.0, Df, Dt, ic_niter, 0, two_pass);
			    
			    this->start_timer();
			    
			    for (int iter = 0; iter < this->niter; iter++)
				ic.clip(intensity, weights, stride);
			    
			    this->stop_timer2(cp);
			}
		    }
		}
	    }
	}
    }
};


int main(int argc, char **argv)
{
    kernel_timing_params params("time-intensity-clipper");
    params.parse_args(argc, argv);

    int nthreads = params.nthreads;
    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<intensity_clipper_timing_thread> (pool, params));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
