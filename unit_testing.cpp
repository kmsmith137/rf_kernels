#include <cmath>
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


timing_thread::timing_thread(const shared_ptr<timing_thread_pool> &pool_, bool pin_to_core) :
    pool(pool_), 
    pinned_to_core(pin_to_core),
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


}  // namespace rf_kernels
