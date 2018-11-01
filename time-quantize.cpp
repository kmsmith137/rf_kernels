#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/quantize.hpp"


using namespace std;
using namespace rf_kernels;


struct quantize_timing_thread : public kernel_timing_thread {
    quantize_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override
    {
	this->allocate();

	int nbits_max = 1;
	int ostride = (nt_chunk * nbits_max) / 8;
	unique_ptr<uint8_t[]> dst(new uint8_t[nfreq * ostride]);

	for (int nbits: { 1 }) {
	    assert(nbits <= nbits_max);

	    stringstream ss;
	    ss << "quantize(nbits=" << nbits << ")";
	    
	    string s = ss.str();
	    const char *cp = s.c_str();
	    
	    quantizer q(nbits);

	    this->start_timer();

	    for (int iter = 0; iter < niter; iter++)
		q.quantize(nfreq, nt_chunk, dst.get(), ostride, weights, stride);

	    this->stop_timer2(cp);
	}
    }
};


int main(int argc, char **argv)
{
    kernel_timing_params params("time-quantize");
    params.parse_args(argc, argv);

    int nthreads = params.nthreads;
    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<quantize_timing_thread> (pool, params));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
