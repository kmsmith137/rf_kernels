#ifndef _RF_KERNELS_UNIT_TESTING_HPP
#define _RF_KERNELS_UNIT_TESTING_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <map>
#include <mutex>
#include <thread>
#include <string>
#include <vector>
#include <random>
#include <cassert>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <condition_variable>

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// RNG helpers


inline double uniform_rand(std::mt19937 &rng)
{
    return std::uniform_real_distribution<>()(rng);
}

inline double uniform_rand(std::mt19937 &rng, double lo, double hi)
{
    return lo + (hi-lo) * uniform_rand(rng);
}

inline ssize_t randint(std::mt19937 &rng, ssize_t lo, ssize_t hi)
{
    assert(lo < hi);
    return std::uniform_int_distribution<>(lo,hi-1)(rng);   // note hi-1 here!
}

inline std::vector<float> uniform_randvec(std::mt19937 &rng, ssize_t n, double lo, double hi)
{
    assert(n > 0);

    std::vector<float> ret(n);
    for (ssize_t i = 0; i < n; i++)
	ret[i] = uniform_rand(rng, lo, hi);

    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// Misc inlines


inline float xsqrt(float x)
{
    assert(x >= 0.0);
    return sqrt(x);
}

inline float maxdiff(const float *v, const float *w, ssize_t n)
{
    assert(n > 0);

    float ret = 0.0;
    for (ssize_t i = 0; i < n; i++)
	ret = std::max(ret, std::abs(v[i]-w[i]));

    return ret;
}


inline float maxdiff(const std::vector<float> &v, const std::vector<float> &w)
{
    assert(v.size() == w.size());
    assert(v.size() > 0);

    return maxdiff(&v[0], &w[0], v.size());
}


inline float maxabs(const float *v, ssize_t n)
{
    assert(n > 0);

    float ret = 0.0;
    for (ssize_t i = 0; i < n; i++)
	ret = std::max(ret, std::abs(v[i]));

    return ret;
}


inline float maxabs(const std::vector<float> &v)
{
    return maxabs(&v[0], v.size());
}


// Does not subtract mean.
inline float weighted_rms_1d(const float *intensity, const float *weights, ssize_t n)
{
    assert(n > 0);

    float wsum = 0.0;
    float wi2sum = 0.0;

    for (ssize_t i = 0; i < n; i++) {
	wsum += weights[i];
	wi2sum += weights[i] * intensity[i] * intensity[i];
    }

    return (wsum > 0) ? sqrt(wi2sum/wsum) : 0.0;
}


inline float weighted_rms_1d(const std::vector<float> &intensity, const std::vector<float> &weights)
{
    assert(intensity.size() == weights.size());
    assert(intensity.size() > 0);
    
    return weighted_rms_1d(&intensity[0], &weights[0], intensity.size());
}


// -------------------------------------------------------------------------------------------------
//
// General-purpose timing thread


class timing_thread_pool {
public:
    const int nthreads;

    timing_thread_pool(int nthreads);

    typedef struct timeval time_t;

    time_t start_timer();
    double stop_timer(const time_t &start_time);
    
    // Helper function called by timing_thread.
    int get_and_increment_thread_id();

protected:
    std::mutex lock;
    std::condition_variable cond0;
    std::condition_variable cond1;
    std::condition_variable cond2;

    double total_dt = 0.0;
    int threads_so_far = 0;

    int ix0 = 0;
    int ix1 = 0;
    int ix2 = 0;
};


class timing_thread {
public:
    const std::shared_ptr<timing_thread_pool> pool;
    const bool pinned_to_core;
    const bool call_warm_up_cpu;
    const int thread_id;
    const int nthreads;

    static void _thread_main(timing_thread *t);

    virtual ~timing_thread() { }

protected:
    timing_thread(const std::shared_ptr<timing_thread_pool> &pool, bool pin_to_core, bool warm_up_cpu=true);

    virtual void thread_body() = 0;

    timing_thread_pool::time_t start_time;
    bool timer_is_running = false;

    // Thread-collective: all threads wait at a barrier, then initialize their local timers.
    void start_timer();

    // Thread-collective: the returned time is the average taken over all threads.
    // If 'name' is non-null, then timing will be announced on thread ID zero.
    double stop_timer(const char *name=nullptr);
};


template<typename T, typename... Args>
std::thread spawn_timing_thread(Args... args)
{
    timing_thread *t = new T(args...);
    return std::thread(timing_thread::_thread_main, t);
}


// -------------------------------------------------------------------------------------------------
//
// lexical_cast: only used as a helper for class argument_parser below.


// Utility routine: converts a string to type T (only a few T's are defined; see lexical_cast.cpp)
// Returns true on success, false on failure
template<typename T> extern bool lexical_cast(const std::string &x, T &ret);

// Also defined in lexical_cast.cpp (for the same values of T)
template<typename T> extern const char *typestr();

// Version of lexical_cast() which throws exception on failure.
template<typename T> inline T lexical_cast(const std::string &x, const char *name="string")
{
    T ret;
    if (lexical_cast(x, ret))
	return ret;
    throw std::runtime_error("couldn't convert " + std::string(name) + "='" + x + "' to " + typestr<T>());
}


// ------------------------------------------------------------------------------------------------
//
// argument_parser


class argument_parser {
public:
    std::vector<std::string> args;
    int nargs = 0;

    void add_boolean_flag(const char *s, bool &flag)
    {
	flag = false;
	this->_add_bflag(s, [&flag]() { flag = true; return true; });
    }

    template<typename T> 
    void add_flag_with_parameter(const char *s, T &val)
    {
	this->_add_pflag(s, [&val](const std::string &v) { return lexical_cast<T> (v,val); });
    }

    template<typename T> 
    void add_flag_with_parameter(const char *s, T &val, bool &flag)
    {
	flag = false;
	this->_add_pflag(s, [&val,&flag](const std::string &v) { flag = true; return lexical_cast<T> (v,val); });
    }

    bool parse_args(int argc, char **argv);

protected:
    std::map<char, std::function<bool()>> _bflags;
    std::map<char, std::function<bool(const std::string &)>> _pflags;
    bool _parsed = false;

    char _get_char(const char *s);
    void _add_bflag(const char *s, std::function<bool()> action);
    void _add_pflag(const char *s, std::function<bool(const std::string &)> action);
    bool _parse_args(int argc, char **argv);
};


// -------------------------------------------------------------------------------------------------
//
// kernel_timing_params: this class parses command-line args assuming syntax
//
//    <prog_name> [-t NTHREADS] [-s STRIDE] [NFREQ] [NT]
//
// kernel_timing_thread: 
//


struct kernel_timing_params {
    const std::string prog_name;

    int nthreads = 1;
    int nfreq = 16384;
    int nt_chunk = 1024;
    int stride = 1032;

    kernel_timing_params(const std::string &prog_name);

    // If there is an error, parse_args() prints an error message and calls exit(1).
    void parse_args(int argc, char **argv);
    
    // Helper for parse_args().
    void usage(const char *msg=nullptr);
};


struct kernel_timing_thread : timing_thread {
    const int niter;
    const int nfreq;
    const int nt_chunk;
    const int stride;

    float *intensity = nullptr;
    float *weights = nullptr;
    
    kernel_timing_thread(const std::shared_ptr<timing_thread_pool> &pool, const kernel_timing_params &params);
    virtual ~kernel_timing_thread();

    // Allocates 'intensity', 'weights'.
    // Note that we don't do this in the constructor, since we want the allocation
    // to be done from the thread's context, for core-pinning reasons.
    void allocate();

    // Prints timing info on thread id 0
    void stop_timer2(const char *kernel_name);
};


}  // namespace rf_kernels

#endif  // _RF_KERNELS_UNIT_TESTING_HPP
