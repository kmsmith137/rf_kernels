#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/online_mask_filler.hpp"

using namespace std;
using namespace rf_kernels;


struct online_mask_filler_timing_thread : public timing_thread {
    const online_mask_filler_params params;
    const int nfreq;
    const int nt_chunk;
    const int stride;

    online_mask_filler_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const online_mask_filler_params &params_, int nfreq_, int nt_chunk_, int stride_) :
	timing_thread(pool_, true),   // pin_to_core=true
	params(params_),
	nfreq(nfreq_),
	nt_chunk(nt_chunk_),
	stride(stride_)
    { }

    virtual ~online_mask_filler_timing_thread() { }

    virtual void thread_body() override 
    {
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
	    online_mask_fill(params, nfreq, nt_chunk, stride, intensity, weights, running_variance, running_weights, rng_state);
	
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


static void usage(const string &msg = string())
{
    cerr << "usage: time-online-mask-filler [-t NTHREADS] [-s STRIDE] [NFREQ] [NT]" << endl;

    if (msg.size() > 0)
	cerr << msg << endl;

    exit(2);
}


int main(int argc, char **argv)
{
    int nthreads = 1;
    int nfreq = 16384;
    int nt_chunk = 1024;
    int stride = 0;

    argument_parser parser;
    parser.add_flag_with_parameter("-t", nthreads);
    parser.add_flag_with_parameter("-s", stride);

    if (!parser.parse_args(argc, argv))
	usage();

    if (parser.nargs > 2)
	usage();
    if (parser.nargs >= 1)
	nfreq = lexical_cast<int> (parser.args[0], "nfreq");
    if (parser.nargs >= 2)
	nfreq = lexical_cast<int> (parser.args[1], "nt_chunk");
    if (stride == 0)
	stride = nt_chunk;

    if (nthreads <= 0)
	usage("Fatal: nthreads must be > 0");
    if (nfreq <= 0)
	usage("Fatal: nfreq must be > 0");
    if (nt_chunk <= 0)
	usage("Fatal: nt_chunk must be > 0");
    if (stride < nt_chunk)
	usage("Fatal: stride must be >= nt_chunk");

    cout << "time-online-mask-filler: nthreads=" << nthreads << ", nfreq=" << nfreq  << ", nt_chunk=" << nt_chunk << ", stride=" << stride << endl;

    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<online_mask_filler_timing_thread> (pool, online_mask_filler_params(), nfreq, nt_chunk, stride));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
