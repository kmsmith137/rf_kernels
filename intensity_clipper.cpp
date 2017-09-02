// FIXME: currently we need to compile a new kernel for every (Df,Dt) pair, where
// Df,Dt are the frequency/time downsampling factors.  Eventually I'd like to 
// improve this by having special kernels to handle the large-Df and large-Dt cases.

#include <unordered_map>

#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/downsample_internals.hpp"
#include "rf_kernels/mean_rms_internals.hpp"
#include "rf_kernels/clipper_internals.hpp"
#include "rf_kernels/intensity_clipper_internals.hpp"
#include "rf_kernels/intensity_clipper.hpp"

using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------


// Inner namespace for the kernel table.
// Must have a different name for each kernel, otherwise gcc is happy but clang has trouble!
namespace intensity_clipper_kernel_table {
#if 0
}; // pacify emacs c-mode
#endif

// kernel(ic, intensity, weights, stride);
using kernel_t = void (*)(const intensity_clipper *ic, const float *, float *, int);
 
// (axis, Df, Dt, two_pass) -> kernel
static unordered_map<array<int,4>, kernel_t> kernel_table;


static kernel_t get_kernel(axis_type axis, int Df, int Dt, bool two_pass)
{
    int t = two_pass ? 1 : 0;
    auto p = kernel_table.find({{axis,Df,Dt,t}});
    
    if (_unlikely(p == kernel_table.end())) {
	stringstream ss;
	ss << "rf_kernels::intensity_clipper: (axis,Df,Dt,two_pass)=("
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

    kernel_table[{{AXIS_TIME,Df,Dt,1}}] = kernel_iclip_Dfsm_Dtsm<float,8,Df,Dt>;
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
    _initializer() { _populate2<2,2>(); }
} _init;

}  // namespace intensity_clipper_kernel_table


// -------------------------------------------------------------------------------------------------


intensity_clipper::intensity_clipper(int nfreq_, int nt_chunk_, axis_type axis_, double sigma_,
				     int Df_, int Dt_, int niter_, double iter_sigma_, bool two_pass_) :
    nfreq(nfreq_),
    nt_chunk(nt_chunk_),
    axis(axis_),
    sigma(sigma_),
    Df(Df_),
    Dt(Dt_),
    niter(niter_),
    iter_sigma(iter_sigma_ ? iter_sigma_ : sigma_),   // note: if iter_sigma=0, then sigma is used instead
    two_pass(two_pass_),
    _f(intensity_clipper_kernel_table::get_kernel(axis_,Df_,Dt_,two_pass_))
{ 
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::intensity_clipper: expected nfreq > 0");

    if (_unlikely(nt_chunk <= 0))
	throw runtime_error("rf_kernels::intensity_clipper: expected nt_chunk > 0");

    if (_unlikely((nfreq % Df) != 0))
	throw runtime_error("rf_kernels::intensity_clipper: expected nfreq to be a multiple of Df");
    
    if (_unlikely((nt_chunk % (8*Dt)) != 0))
	throw runtime_error("rf_kernels::intensity_clipper: expected nt_chunk to be a multiple of 8*Dt");

    if (_unlikely(sigma < 1.0))
	throw runtime_error("rf_kernels::intensity_clipper: expected sigma >= 1.0");

    if (_unlikely(iter_sigma < 1.0))
	throw runtime_error("rf_kernels::intensity_clipper: expected iter_sigma >= 1.0");

    if (_unlikely(niter < 1))
	throw runtime_error("rf_kernels::intensity_clipper: expected niter >= 1");

    this->nfreq_ds = xdiv(nfreq, Df);
    this->nt_ds = xdiv(nt_chunk, Dt);

    // Depending on the arguments to the intensity_clipper, we may or may not need to allocate
    // temporary buffers for downsampled intensity and weights.
    //
    // FIXME: the details of this logic are opaque and depend on chasing through rf_kernels/*.hpp!
    // It would be nice to have comments in these files which make it more transparent.

    // FIXME revisit this.
    // if ((Df==1) && (Dt==1))
    //return;   // no allocation necessary

    int nds = 0;

    if (axis == AXIS_FREQ)
	nds = nfreq_ds * 8;
    else if (axis == AXIS_TIME)
	nds = nt_ds;
    else if (axis == AXIS_NONE)
	nds = nfreq_ds * nt_ds;
    else
	throw runtime_error("rf_kernels internal error: bad axis in intensity_clipper constructor");

    this->tmp_i = aligned_alloc<float> (nds);

    if ((niter == 1) && !two_pass)
	return;  // no ds_weights necessary
    
    this->tmp_w = aligned_alloc<float> (nds);
}


intensity_clipper::~intensity_clipper()
{
    free(tmp_i);
    free(tmp_w);
    tmp_i = tmp_w = nullptr;
}


void intensity_clipper::clip(const float *intensity, float *weights, int stride)
{
    if (_unlikely(!intensity || !weights))
	throw runtime_error("rf_kernels: null pointer passed to intensity_clipper::clip()");

    if (_unlikely(abs(stride) < nt_chunk))
	throw runtime_error("rf_kernels::intensity_clipper: stride is too small");

    this->_f(this, intensity, weights, stride);
}


}  // namespace rf_kernels
