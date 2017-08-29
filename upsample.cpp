#include <unordered_map>

#include "rf_kernels/internals.hpp"
#include "rf_kernels/upsample.hpp"
#include "rf_kernels/upsample_internals.hpp"

using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// Usage: kernel(nfreq_in, nt_in, dst, dstride, src, sstride, w_cutoff, Df, Dt)
using upsampling_kernel_t = void (*)(int, int, float *, int, const float *, int, float, int, int);


// -------------------------------------------------------------------------------------------------
//
// global kernel table


static unordered_map<array<int,2>, upsampling_kernel_t> global_kernel_table;   // (Df,Dt) -> kernel


inline void _bad_Df_Dt(int Df, int Dt)
{
    stringstream ss;
    ss << "rf_kernels::weighted_upsampler: (Df,Dt)=(" << Df << "," << Dt << ") is not supported";
    throw runtime_error(ss.str());
}


inline upsampling_kernel_t get_kernel(int Df, int Dt)
{
    if (Df > 8) {
	if (_unlikely(Df % 8 != 0))
	    _bad_Df_Dt(Df,Dt);
	Df = 16;
    }

    if (Dt > 8) {
	if (_unlikely(Dt % 8 != 0))
	    _bad_Df_Dt(Df,Dt);
	Dt = 16;
    }

    auto p = global_kernel_table.find({{Df,Dt}});
    
    if (_unlikely(p == global_kernel_table.end()))
	_bad_Df_Dt(Df, Dt);

    return p->second;
}


// -------------------------------------------------------------------------------------------------
//
// Populating global kernel table


template<int Df, int Dt> struct kernel
{
    static upsampling_kernel_t get() { return kernel_upsample_weights_Df_Dt<Df,Dt>; }
};

template<int Df> struct kernel<Df,16>
{
    static upsampling_kernel_t get() { return kernel_upsample_weights_Df<Df>; }
};

template<int Dt> struct kernel<16,Dt>
{
    static upsampling_kernel_t get() { return kernel_upsample_weights_Dt<Dt>; }
};

template<> struct kernel<16,16>
{
    static upsampling_kernel_t get() { return kernel_upsample_weights; }
};


template<int Df, int Dt, typename enable_if<(Dt==0),int>::type=0>
inline void _populate1() { }

template<int Df, int Dt, typename enable_if<(Dt>0),int>::type=0>
inline void _populate1()
{
    _populate1<Df,(Dt/2)> ();
    global_kernel_table[{{Df,Dt}}] = kernel<Df,Dt>::get();
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
	X() { _populate2<16,16>(); }
    } x;
}


// -------------------------------------------------------------------------------------------------


weight_upsampler::weight_upsampler(int Df_, int Dt_) :
    Df(Df_),
    Dt(Dt_),
    _f(get_kernel(Df_,Dt_))
{ }


void weight_upsampler::upsample(int nfreq_in, int nt_in, float *out, int ostride, const float *in, int istride, float w_cutoff)
{
    if (_unlikely(nfreq_in <= 0))
	throw runtime_error("rf_kernels::weighted_upsample: expected nfreq_in > 0");
    if (_unlikely(nt_in <= 0))
	throw runtime_error("rf_kernels::weighted_upsample: expected nt_in > 0");
    if (_unlikely(nt_in % 8))
	throw runtime_error("rf_kernels::weighted_upsample: expected nt_in divisible by 8");
    if (_unlikely(!out || !in))
	throw runtime_error("rf_kernels::weighted_upsample: null pointer passed to upsample()");
    if (_unlikely(abs(ostride) < Dt*nt_in))
	throw runtime_error("rf_kernels::weighted_upsample: ostride is too small");
    if (_unlikely(abs(istride) < nt_in))
	throw runtime_error("rf_kernels::weighted_upsample: istride is too small");
    if (_unlikely(w_cutoff < 0.0))
	throw runtime_error("rf_kernels::weighted_upsample: expected w_cutoff >= 0");

    this->_f(nfreq_in, nt_in, out, ostride, in, istride, w_cutoff, this->Df, this->Dt);
}



}   // namespace rf_kernels
