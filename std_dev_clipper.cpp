// FIXME: currently we need to compile a new kernel for every (Df,Dt) pair, where
// Df,Dt are the frequency/time downsampling factors.  Eventually I'd like to 
// improve this by having special kernels to handle the large-Df and large-Dt cases.

#include <unordered_map>

#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/downsample_internals.hpp"
#include "rf_kernels/mean_rms_internals.hpp"
#include "rf_kernels/std_dev_clipper_internals.hpp"
#include "rf_kernels/std_dev_clipper.hpp"

using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// Externally-linkable helper function, declared "extern" in kernels/std_dev_clippers.hpp.
// If the arguments change here, then declaration should be changed there as well!


void clip_1d(int n, float *tmp_sd, int *tmp_valid, double sigma)
{
#if 0
    cerr << "clip_1d: [";
    for (int i = 0; i < n; i++) {
	if (tmp_valid[i])
	    cerr << " " << tmp_sd[i];
	else
	    cerr << " -";
    }
    cerr << " ]\n";
#endif

    float acc0 = 0.0;
    float acc1 = 0.0;
    
    for (int i = 0; i < n; i++) {
	if (tmp_valid[i]) {
	    acc0 += 1.0;
	    acc1 += tmp_sd[i];
	}
    }
    
    if (acc0 < 1.5) {
	memset(tmp_valid, 0, n * sizeof(int));
	return;
    }
    
    float mean = acc1 / acc0;
    float acc2 = 0.0;
    
    for (int i = 0; i < n; i++)
	if (tmp_valid[i])
	    acc2 += square(tmp_sd[i] - mean);
    
    float stdv = sqrtf(acc2/acc0);
    float thresh = sigma * stdv;

    for (int i = 0; i < n; i++) {
	if (fabs(tmp_sd[i] - mean) >= thresh)
	    tmp_valid[i] = 0;
    }
}

// -------------------------------------------------------------------------------------------------


// Inner namespace for the kernel table
// Must have a different name for each kernel, otherwise gcc is happy but clang has trouble!
namespace std_dev_clipper_table {
#if 0
}; // pacify emacs c-mode
#endif

// kernel(intensity, weights, nfreq, nt, stride, sigma, out_sd, out_valid, ds_int, ds_wt)
using kernel_t = void (*)(const float *, float *, int, int, int, double, float *, int *, float *, float *);

// (axis, Df, Dt, two_pass) -> kernel
static unordered_map<array<int,4>, kernel_t> kernel_table;


static kernel_t get_kernel(axis_type axis, int Df, int Dt, bool two_pass)
{
    int t = two_pass ? 1 : 0;
    auto p = kernel_table.find({{axis,Df,Dt,t}});
    
    if (_unlikely(p == kernel_table.end())) {
	stringstream ss;
	ss << "rf_kernels::std_dev_clipper: (axis,Df,Dt,two_pass)=("
	   << axis << "," << Df << "," << Dt << "," << t << ") is invalid or unimplemented";
	throw runtime_error(ss.str());
    }

    return p->second;
}


template<int Df, int Dt, typename enable_if<(Dt==0),int>::type = 0>
inline void _populate1() { }

template<int Df, int Dt, typename enable_if<(Dt>0),int>::type = 0>
inline void _populate1()
{
    _populate1<Df,(Dt/2)> ();

    kernel_table[{{AXIS_FREQ,Df,Dt,false}}] = _kernel_std_dev_clip_freq_axis<float,8,Df,Dt,false>;
    kernel_table[{{AXIS_TIME,Df,Dt,false}}] = _kernel_std_dev_clip_time_axis<float,8,Df,Dt,false>;
    kernel_table[{{AXIS_FREQ,Df,Dt,true}}] = _kernel_std_dev_clip_freq_axis<float,8,Df,Dt,true>;
    kernel_table[{{AXIS_TIME,Df,Dt,true}}] = _kernel_std_dev_clip_time_axis<float,8,Df,Dt,true>;
}


template<int Df, int Dt, typename enable_if<(Df==0),int>::type = 0>
inline void _populate2() { }

template<int Df, int Dt, typename enable_if<(Df>0),int>::type = 0>
inline void _populate2()
{
    _populate2<(Df/2),Dt> ();
    _populate1<Df,Dt> ();
}

struct _initializer {
    _initializer() { _populate2<16,16>(); }
} _init;


}  // namespace std_dev_clipper_table


// -------------------------------------------------------------------------------------------------


std_dev_clipper::std_dev_clipper(int nfreq_, int nt_chunk_, axis_type axis_, double sigma_, int Df_, int Dt_, bool two_pass_) :
    nfreq(nfreq_),
    nt_chunk(nt_chunk_),
    axis(axis_),
    Df(Df_),
    Dt(Dt_),
    sigma(sigma_),
    two_pass(two_pass_),
    _f(std_dev_clipper_table::get_kernel(axis_,Df_,Dt_,two_pass_))
{
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::std_dev_clipper: expected nfreq > 0");

    if (_unlikely(nt_chunk <= 0))
	throw runtime_error("rf_kernels::std_dev_clipper: expected nt_chunk > 0");

    if (_unlikely(nfreq % Df))
	throw runtime_error("rf_kernels::std_dev_clipper: expected nfreq to be a multiple of Df");

    if (_unlikely(nt_chunk % (8*Dt)))
	throw runtime_error("rf_kernels::std_dev_clipper: expected nfreq to be a multiple of 8*Dt");

    if (_unlikely(sigma < 1.0))
	throw runtime_error("rf_kernels::std_dev_clipper: expected sigma >= 1");

    // The sizes of the temporary buffers needed for the std_dev_clipper depend on its arguments.
    //
    // FIXME: the details of this logic are opaque and depend on chasing through kernels/*.hpp!
    // It would be nice to have comments in these files which make it more transparent.

    int sd_nalloc = (axis == AXIS_FREQ) ? (nt_chunk/Dt) : (nfreq/Df);

    this->sd = aligned_alloc<float> (sd_nalloc);
    this->sd_valid = aligned_alloc<int> (sd_nalloc);

    if (two_pass && ((Df > 1) || (Dt > 1))) {
	this->ds_intensity = aligned_alloc<float> ((nfreq*nt_chunk) / (Df*Dt));
	this->ds_weights = aligned_alloc<float> ((nfreq*nt_chunk) / (Df*Dt));
    }
}


std_dev_clipper::~std_dev_clipper()
{
    free(sd);
    free(sd_valid);
    free(ds_intensity);
    free(ds_weights);

    sd = ds_intensity = ds_weights = nullptr;
    sd_valid = nullptr;
}
   

void std_dev_clipper::clip(const float *intensity, float *weights, int stride)
{
    if (_unlikely(!intensity || !weights))
	throw runtime_error("rf_kernels: null pointer passed to std_dev_clipper::clip()");
    if (_unlikely(abs(stride) < nt_chunk))
	throw runtime_error("rf_kernels::std_dev_clipper: stride is too small");

    this->_f(intensity, weights, nfreq, nt_chunk, stride, sigma, sd, sd_valid, ds_intensity, ds_weights);
}


}  // namespace rf_kernels
