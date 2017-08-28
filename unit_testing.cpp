#include <cmath>
#include <cstring>
#include <climits>
#include <unistd.h>

#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"

using namespace std;

namespace rf_kernels {
#if 0
}  // emacs pacifier
#endif


static void pin_current_thread_to_core(int core_id)
{
#ifdef __APPLE__
    if (core_id == 0)
	cerr << "warning: pinning threads to cores is not implemented in osx\n";
    return;
#else
    int hwcores = std::thread::hardware_concurrency();
    
    if ((core_id < 0) || (core_id >= hwcores))
	throw runtime_error("pin_thread_to_core: core_id=" + to_string(core_id) + " is out of range (hwcores=" + to_string(hwcores) + ")");

    pthread_t thread = pthread_self();

    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(core_id, &cs);

    int err = pthread_setaffinity_np(thread, sizeof(cs), &cs);
    if (err)
        throw runtime_error("pthread_setaffinity_np() failed");
#endif
}


void warm_up_cpu()
{
    // A throwaway computation which uses the CPU for ~10^9
    // clock cycles.  The details (usleep, xor) are to prevent the
    // compiler from optimizing it out!
    //
    // Empirically, this makes timing results more stable (without it,
    // the CPU seems to run slow for the first ~10^9 cycles or so.)

    long n = 0;
    for (long i = 0; i < 1000L * 1000L * 1000L; i++)
	n += (i ^ (i-1));
    usleep(n % 2);
}


// -------------------------------------------------------------------------------------------------
//
// timing_thread_pool


timing_thread_pool::timing_thread_pool(int nthreads_) :
    nthreads(nthreads_)
{ 
    if (nthreads <= 0)
	throw runtime_error("timing_thread_pool constructor called with nthreads <= 0");
}


timing_thread_pool::time_t timing_thread_pool::start_timer()
{
    unique_lock<mutex> l(lock);

    ix0++;
    if (ix0 == nthreads) {
	ix1 = ix2 = 0;
	cond0.notify_all();
    }
    while (ix0 < nthreads)
	cond0.wait(l);

    time_t start_time;
    gettimeofday(&start_time, NULL);
    return start_time;
}


double timing_thread_pool::stop_timer(const time_t &start_time)
{
    time_t end_time;
    gettimeofday(&end_time, NULL);

    double local_dt = (end_time.tv_sec - start_time.tv_sec) + 1.0e-6 * (end_time.tv_usec - start_time.tv_usec);

    unique_lock<mutex> l(this->lock);
    total_dt += local_dt;

    ix1++;
    if (ix1 == nthreads) {
	ix0 = ix2 = 0;
	cond1.notify_all();
    }
    while (ix1 < nthreads)
	cond1.wait(l);

    double ret = total_dt / nthreads;

    ix2++;
    if (ix2 == nthreads) {
	total_dt = 0.0;
	ix0 = ix1 = 0;
	cond2.notify_all();
    }
    while (ix2 < nthreads)
	cond2.wait(l);

    return ret;
}


int timing_thread_pool::get_and_increment_thread_id()
{
    lock_guard<mutex> l(lock);
    return threads_so_far++;
}


// -------------------------------------------------------------------------------------------------
//
// timing_thread


timing_thread::timing_thread(const shared_ptr<timing_thread_pool> &pool_, bool pin_to_core, bool warm_up_cpu_) :
    pool(pool_), 
    pinned_to_core(pin_to_core),
    call_warm_up_cpu(warm_up_cpu_),
    thread_id(pool_->get_and_increment_thread_id()),
    nthreads(pool_->nthreads)
{ }


// static member function
void timing_thread::_thread_main(timing_thread *t)
{
    // Ensure delete(t) is called
    auto p = unique_ptr<timing_thread> (t);

    if (t->pinned_to_core)
	pin_current_thread_to_core(t->thread_id);

    // Call after pinning thread
    if (t->call_warm_up_cpu)
	warm_up_cpu();

    t->thread_body();
}


void timing_thread::start_timer()
{
    if (timer_is_running)
	throw runtime_error("double call to timing_thread::start_timer(), without call to stop_timer() in between");

    this->timer_is_running = true;
    this->start_time = pool->start_timer();
}


double timing_thread::stop_timer(const char *name)
{
    if (!timer_is_running)
	throw runtime_error("timing_thread::stop_timer() called without calling start_timer(), or double call to stop_timer()");

    this->timer_is_running = false;

    double ret = pool->stop_timer(start_time);

    if (name && !thread_id)
	cout << (string(name) + ": " + to_string(ret) + " seconds\n");

    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// lexical_cast


template<> const char *typestr<string>()    { return "string"; }
template<> const char *typestr<long>()      { return "long"; }
template<> const char *typestr<int>()       { return "int"; }
template<> const char *typestr<double>()    { return "double"; }
template<> const char *typestr<float>()     { return "float"; }
template<> const char *typestr<uint16_t>()  { return "uint16_t"; }
template<> const char *typestr<bool>()      { return "bool"; }


// trivial case: convert string -> string
template<> bool lexical_cast(const string &x, string &ret) { ret = x; return true;}


inline bool is_all_spaces(const char *s)
{
    if (!s)
	throw runtime_error("fatal: NULL pointer passed to is_all_spaces()");

    for (;;) {
	if (!*s)
	    return true;
	if (!isspace(*s))
	    return false;
	s++;
    }
}


template<> bool lexical_cast(const string &x, long &ret)
{ 
    const char *ptr = x.c_str();
    char *endptr = NULL;

    ret = strtol(ptr, &endptr, 10);
    return (endptr != ptr) && (ret != LONG_MIN) && (ret != LONG_MAX) && is_all_spaces(endptr);
}


template<> bool lexical_cast(const string &x, int &ret)
{
    long retl;

    if (!lexical_cast(x, retl))
	return false;
    if ((sizeof(int) != sizeof(long)) && ((retl < INT_MIN) || (retl > INT_MAX)))
	return false;

    ret = retl;
    return true;
}


template<> bool lexical_cast(const string &x, uint16_t &ret)
{
    long retl;

    if (!lexical_cast(x, retl))
	return false;
    if ((retl < 0) || (retl > 65535))
	return false;

    ret = retl;
    return true;
}


template<> bool lexical_cast(const string &x, double &ret)
{ 
    const char *ptr = x.c_str();
    char *endptr = NULL;

    ret = strtod(ptr, &endptr);
    return (endptr != ptr) && (ret != -HUGE_VAL) && (ret != HUGE_VAL) && is_all_spaces(endptr);
}


template<> bool lexical_cast(const string &x, float &ret)
{ 
    const char *ptr = x.c_str();
    char *endptr = NULL;

    ret = strtof(ptr, &endptr);
    return (endptr != ptr) && (ret != -HUGE_VALF) && (ret != HUGE_VALF) && is_all_spaces(endptr);
}

template<> bool lexical_cast(const string &x, bool &ret)
{
    const char *ptr = x.c_str();

    if (!strcasecmp(ptr,"t") || !strcasecmp(ptr,"true"))
	ret = true;
    else if (!strcasecmp(ptr,"f") || !strcasecmp(ptr,"false"))
	ret = false;
    else
	return false;

    return true;
}


// -------------------------------------------------------------------------------------------------
//
// argument_parser


char argument_parser::_get_char(const char *s)
{
    if ((s[0] != '-') || (s[1] == 0) || (s[2] != 0))
	throw runtime_error("argument_parser: expected argument \"" + string(s) + "\" to have form \"-X\"");

    char c = s[1];

    if ((this->_bflags.find(c) != this->_bflags.end()) || (this->_pflags.find(c) != this->_pflags.end()))
	throw runtime_error("argument_parser: multiple \"" + string(s) + "\"entries");

    return c;
}


void argument_parser::_add_bflag(const char *s, std::function<bool()> action)
{
    char c = _get_char(s);
    _bflags[c] = action;
}


void argument_parser::_add_pflag(const char *s, std::function<bool(const string &)> action)
{
    char c = _get_char(s);
    _pflags[c] = action;
}


bool argument_parser::parse_args(int argc, char **argv)
{
    bool ret = _parse_args(argc, argv);
    
    if (ret)
	this->nargs = args.size();
    else {
	this->nargs = 0;
	this->args.clear();
    }

    return ret;
}


bool argument_parser::_parse_args(int argc, char **argv)
{
    if (_parsed)
	throw runtime_error("double call to argument_parser::parse_args()");
	    
    this->_parsed = true;

    // Note loop starts at i=1
    for (int i = 1; i < argc; i++) {
	if (argv[i][0] != '-') {
	    this->args.push_back(argv[i]);
	    continue;
	}

	if (!strcmp(argv[i], "--")) {
	    for (int j = i; j < argc; j++)
		this->args.push_back(argv[j]);
	    return true;
	}

	int n = strlen(argv[i]);

	if (n <= 1) {
	    this->args.clear();
	    return false;
	}

	if ((n == 2) && (i+1 < argc)) {
	    auto p = this->_pflags.find(argv[i][1]);

	    if (p != this->_pflags.end()) {
		if (!p->second(argv[i+1]))
		    return false;
		i++;  // extra advance
		continue;
	    }
	    
	    // fallthrough here intentional...
	}

	for (int j = 1; j < n; j++) {
	    auto p = this->_bflags.find(argv[i][j]);
	    
	    if (p == this->_bflags.end())
		return false;
	    if (!p->second())
		return false;
	}
    }

    return true;
}


// -------------------------------------------------------------------------------------------------
//
// kernel_timing_params


kernel_timing_params::kernel_timing_params(const string &prog_name_)
    : prog_name(prog_name_)
{ }


void kernel_timing_params::usage(const char *msg)
{
    cerr << "usage: " << prog_name << " [-t NTHREADS] [-s STRIDE] [NFREQ] [NT]" << endl;

    if (msg != nullptr)
	cerr << msg << endl;
    
    exit(2);
}


void kernel_timing_params::parse_args(int argc, char **argv)
{
    argument_parser p;
    p.add_flag_with_parameter("-t", this->nthreads);
    p.add_flag_with_parameter("-s", this->stride);

    if (!p.parse_args(argc, argv))
	usage();
    
    if (p.nargs > 2)
	usage();
    if (p.nargs >= 1)
	nfreq = lexical_cast<int> (p.args[0], "nfreq");
    if (p.nargs >= 2)
	nfreq = lexical_cast<int> (p.args[1], "nt_chunk");
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

    cout << prog_name << ": nthreads=" << nthreads << ", nfreq=" << nfreq  << ", nt_chunk=" << nt_chunk << ", stride=" << stride << endl;
}


// -------------------------------------------------------------------------------------------------
//
// kernel_timing_thread


inline int _assign_niter(const kernel_timing_params &params)
{
    rf_assert(params.nfreq > 0);
    rf_assert(params.nt_chunk > 0);

    double x = 1.0e9 / double(params.nfreq) / double(params.nt_chunk);
    return int(x) + 1;
}

kernel_timing_thread::kernel_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params) :
    timing_thread(pool_, true, true),    // (pin_to_core, warm_up_cpu) = (true, true)
    niter(_assign_niter(params)),
    nfreq(params.nfreq),
    nt_chunk(params.nt_chunk),
    stride(params.stride)
{ 
    // Note: we don't allocate 'intensity' and 'weights' in the kernel_timing_thread constructor,
    // since this would allocate from the "master" thread context.  Instead, these arrays are
    // allocated in kernel-timing_thread::allocate(), which is called from the timing thread context.
}


kernel_timing_thread::~kernel_timing_thread()
{
    free(intensity);
    free(weights);
    intensity = weights = nullptr;
}


void kernel_timing_thread::allocate()
{
    if (!intensity)
	intensity = aligned_alloc<float> (nfreq * stride);
    if (!weights)
	weights = aligned_alloc<float> (nfreq * stride);

    for (int i = 0; i < nfreq*stride; i++)
	weights[i] = 1.0;

    // intensities will be zero, since aligned_alloc() zeroes the buffer.
}


void kernel_timing_thread::stop_timer2(const char *kernel_name)
{
    double dt = this->stop_timer();
    ssize_t nsamples = ssize_t(niter) * ssize_t(nfreq) * ssize_t(nt_chunk);
    
    if (thread_id == 0) {
	cout << kernel_name << ": " << niter << " iterations,"
	     << " total time " << dt << " sec"
	     << " (" << (dt/niter) << " sec/iteration,"
	     << " " << (1.0e9 * dt / nsamples) << " ns/sample)" << endl;
    }
}


}  // namespace rf_kernels
