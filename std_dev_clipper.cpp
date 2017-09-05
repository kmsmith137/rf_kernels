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


// Inner namespace for the kernel table
// Must have a different name for each kernel, otherwise gcc is happy but clang has trouble!
namespace std_dev_clipper_table {
#if 0
}; // pacify emacs c-mode
#endif

// kernel(sd, intensity, weights, stride)
using kernel_t = void (*)(std_dev_clipper *sd, const float *, float *, int);

// (axis, Df, Dt, two_pass) -> kernel
static unordered_map<array<int,4>, kernel_t> kernel_table;


static kernel_t get_kernel(axis_type axis, int Df, int Dt, bool two_pass)
{
    if ((Df > 8) && (Df % 8 == 0))
	Df = 16;
    
    if ((Dt > 8) && (Dt % 8 == 0))
	Dt = 16;

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
    
    kernel_table[{{AXIS_TIME,Df,Dt,0}}] = kernel_std_dev_clipper_taxis<float,8,Df,Dt,false>;
    kernel_table[{{AXIS_FREQ,Df,Dt,0}}] = kernel_std_dev_clipper_faxis<float,8,Df,Dt,false>;

    kernel_table[{{AXIS_TIME,Df,Dt,1}}] = kernel_std_dev_clipper_taxis<float,8,Df,Dt,true>;
    kernel_table[{{AXIS_FREQ,Df,Dt,1}}] = kernel_std_dev_clipper_faxis<float,8,Df,Dt,true>;
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

    // Assumed in kernel_std_dev_clipper_faxis(), see comment in rf_kernels/std_dev_clipper_internals.hpp.
    if (_unlikely(nfreq % 8))
	throw runtime_error("rf_kernels::std_dev_clipper: expected nfreq to be a multiple of 8");

    if (_unlikely(nt_chunk % (8*Dt)))
	throw runtime_error("rf_kernels::std_dev_clipper: expected nfreq to be a multiple of 8*Dt");

    if (_unlikely(sigma < 1.0))
	throw runtime_error("rf_kernels::std_dev_clipper: expected sigma >= 1");

    this->nfreq_ds = xdiv(nfreq, Df);
    this->nt_ds = xdiv(nt_chunk, Dt);

    if (axis == AXIS_FREQ) {
	this->ntmp_wi = 8 * nfreq_ds;
	this->ntmp_v = nt_ds;
    }
    else if (axis == AXIS_TIME) {
	this->ntmp_wi = nt_ds;
	this->ntmp_v = nfreq_ds;
    }
    else 
	throw runtime_error("rf_kernels: internal error: bad axis in std_dev_clipper constructor");

    this->tmp_i = aligned_alloc<float> (ntmp_wi);
    this->tmp_w = aligned_alloc<float> (ntmp_wi);
    this->tmp_v = aligned_alloc<float> (ntmp_v);
}


std_dev_clipper::~std_dev_clipper()
{
    free(tmp_i);
    free(tmp_w);
    free(tmp_v);

    tmp_i = tmp_w = tmp_v = nullptr;
}
   

void std_dev_clipper::clip(const float *intensity, float *weights, int stride)
{
    if (_unlikely(!intensity || !weights))
	throw runtime_error("rf_kernels: null pointer passed to std_dev_clipper::clip()");
    if (_unlikely(abs(stride) < nt_chunk))
	throw runtime_error("rf_kernels::std_dev_clipper: stride is too small");

    this->_f(this, intensity, weights, stride);
}


// Scalar helper called by kernel.
//
// Note that std_dev_clipper::_clip_1d() isn't unit-tested anywhere (in particular,
// test-std-dev-clippers is a circular test of _clip_1d()).  This is because I decided
// it was so simple that a reference implementation would just be equivalent.  If you
// modify it, be careful!

void std_dev_clipper::_clip_1d()
{
#if 0
    cout << "clip_1d top: [";
    for (int i = 0; i < ntmp_v; i++)
	cout << " " << tmp_v[i];
    cout << " ]\n";
#endif

    float acc0 = 0.0;
    float acc1 = 0.0;
    
    for (int i = 0; i < ntmp_v; i++) {
	if (tmp_v[i] > 0.0) {
	    acc0 += 1.0;
	    acc1 += tmp_v[i];
	}
    }
    
    if (acc0 < 1.5) {
	memset(tmp_v, 0, ntmp_v * sizeof(float));
	return;
    }
    
    float mean = acc1 / acc0;
    float acc2 = 0.0;
    
    for (int i = 0; i < ntmp_v; i++)
	if (tmp_v[i] > 0.0)
	    acc2 += square(tmp_v[i] - mean);
    
    float stdv = sqrtf(acc2/acc0);
    float thresh = sigma * stdv;

    for (int i = 0; i < ntmp_v; i++) {
	if (fabs(tmp_v[i] - mean) >= thresh)
	    tmp_v[i] = 0;
    }

#if 0
    cout << "clip_1d bottom: [";
    for (int i = 0; i < ntmp_v; i++)
	cout << " " << tmp_v[i];
    cout << " ]\n";
#endif
}


}  // namespace rf_kernels
