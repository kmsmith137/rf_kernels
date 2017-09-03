#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/std_dev_clipper.hpp"


using namespace std;
using namespace rf_kernels;


struct std_dev_clipper_timing_thread : public kernel_timing_thread {
    std_dev_clipper_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override
    {
	this->allocate();

	// The std_dev_clipper's running time depends on what fraction of data
	// actually gets masked!  For the test, I decided to fill 'intensity'
	// with random values, set all weights to 1, clip at 1-sigma, and
	// reset all weights to 1 between iterations of the timing loop.

	std::random_device rd;
	std::mt19937 rng(rd());
	for (int i = 0; i < nfreq * stride; i++)
	    intensity[i] = uniform_rand(rng, -1.0, 1.0);

	for (int Df: { 1, 2, 4, 8, 16, 64, 256 }) {
	    for (int Dt: { 1, 2, 4, 8, 16 }) {
		// for (axis_type axis: { AXIS_FREQ, AXIS_TIME })  {
		for (axis_type axis: { AXIS_TIME })  {
		    // for (bool two_pass: {false,true}) {
		    for (bool two_pass: {true}) {
			stringstream ss;
			ss << "std_dev_clipper(axis=" << axis << ",Df=" << Df << ",Dt=" << Dt << ",two_pass=" << two_pass << ")";
			    
			string s = ss.str();
			const char *cp = s.c_str();

			double sigma = 1.0;
			std_dev_clipper sd(nfreq, nt_chunk, axis, sigma, Df, Dt, two_pass);
			
			this->start_timer();
			
			for (int iter = 0; iter < this->niter; iter++) {
			    // As described above, we reset all weights to 1 between
			    // iterations of the timing loop.  We pause the timer so
			    // that this resetting isn't included in the timing result.
			    
			    this->pause_timer();
			    
			    for (int ifreq = 0; ifreq < nfreq; ifreq++)
				for (int it = 0; it < nt_chunk; it++)
				    weights[ifreq*stride + it] = 1.0;

			    this->unpause_timer();
			    
			    sd.clip(intensity, weights, stride);
			}
			
			this->stop_timer2(cp);
		    }
		}
	    }
	}
    }
};


int main(int argc, char **argv)
{
    kernel_timing_params params("time-std-dev-clipper");
    params.parse_args(argc, argv);

    int nthreads = params.nthreads;
    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<std_dev_clipper_timing_thread> (pool, params));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
