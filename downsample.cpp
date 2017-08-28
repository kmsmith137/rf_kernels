#include <unordered_map>

#include "rf_kernels/internals.hpp"
#include "rf_kernels/downsample.hpp"
#include "rf_kernels/downsample_internals.hpp"


using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// Usage: kernel(nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride, Df, Dt)
using downsampling_kernel_t = void (*)(int, int, float *, float *, int, const float *, const float *, int, int, int);


// FIXME this wrapper will go away soon
template<int Df, int Dt>
inline void kernel_wi_downsample_Df_Dt(int nfreq_out, int nt_out, float *out_i, float *out_w, int ostride,
				       const float *in_i, const float *in_w, int istride, int Df_, int Dt_)
{
    _kernel_downsample_2d<float,8,Df,Dt> (out_i, out_w, ostride, in_i, in_w, nfreq_out * Df, nt_out * Dt, istride);
}


// -------------------------------------------------------------------------------------------------
//
// global kernel table


static unordered_map<array<int,2>, downsampling_kernel_t> global_kernel_table;   // (Df,Dt) -> kernel


inline void _bad_Df_Dt(int Df, int Dt)
{
    stringstream ss;
    ss << "rf_kernels::wi_downsampler: (Df,Dt)=(" << Df << "," << Dt << ") is not supported";
    throw runtime_error(ss.str());
}


inline downsampling_kernel_t get_kernel(int Df, int Dt)
{
    auto p = global_kernel_table.find({Df,Dt});
    
    if (_unlikely(p == global_kernel_table.end()))
	_bad_Df_Dt(Df, Dt);

    return p->second;
}


template<int Df, int Dt, typename enable_if<(Dt==0),int>::type=0>
inline void _populate1() { }

template<int Df, int Dt, typename enable_if<(Dt>0),int>::type=0>
inline void _populate1()
{
    _populate1<Df,(Dt/2)> ();
    global_kernel_table[{Df,Dt}] = kernel_wi_downsample_Df_Dt<Df,Dt>;
}


template<int Df, int Dt, typename enable_if<(Df==0),int>::type=0>
inline void _populate2() { }

// Called for (Df,Dt)=(*,8).
template<int Df, int Dt, typename enable_if<(Df>0),int>::type=0>
inline void _populate2()
{
    _populate2<(Df/2),Dt> ();
    _populate1<Df,Dt> ();
}


namespace {
    struct X {
	X() { _populate2<256,32>(); }
    } x;
}


// -------------------------------------------------------------------------------------------------


wi_downsampler::wi_downsampler(int Df_, int Dt_) :
    Df(Df_),
    Dt(Dt_),
    _f(get_kernel(Df_,Dt_))
{ }
    

void wi_downsampler::downsample(int nfreq_out, int nt_out, float *out_i, float *out_w, int ostride, const float *in_i, const float *in_w, int istride)
{
    if (_unlikely(nfreq_out <= 0))
	throw runtime_error("rf_kernels::wi_downsampler: expected nfreq_out > 0");
    
    if (_unlikely(nt_out <= 0))
	throw runtime_error("rf_kernels::wi_downsampler: expected nt_out > 0");
    
    if (_unlikely(nt_out % 8))
	throw runtime_error("rf_kernels::wi_downsampler: expected nt_out divisible by 8");

    if (_unlikely(abs(istride) < Dt * nt_out))
	throw runtime_error("rf_kernels::wi_downsampler: istride is too small");

    if (_unlikely(abs(ostride) < nt_out))
	throw runtime_error("rf_kernels::wi_downsampler: ostride is too small");
    
    if (_unlikely(!out_i || !out_w || !in_i || !in_w))
	throw runtime_error("rf_kernels: null pointer passed to wi_downsampler::downsample()");

    this->_f(nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride, Df, Dt);
}


}  // namespace rf_kernels
