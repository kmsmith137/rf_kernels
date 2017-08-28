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


static unordered_map<array<int,2>, upsampling_kernel_t>  global_kernel_table_Df_Dt;   // (Df,Dt) -> kernel
static unordered_map<int, upsampling_kernel_t>           global_kernel_table_Df;      // (Df) -> kernel
static unordered_map<int, upsampling_kernel_t>           global_kernel_table_Dt;      // (Dt) -> kernel


inline void _bad_Df_Dt(int Df, int Dt)
{
    stringstream ss;
    ss << "rf_kernels::weighted_upsampler: (Df,Dt)=(" << Df << "," << Dt << ") is not supported";
    throw runtime_error(ss.str());
}


template<typename T>
inline upsampling_kernel_t _get_kernel(const unordered_map<T,upsampling_kernel_t> &t, const T &key, int Df, int Dt)
{
    auto p = t.find(key);
    if (_unlikely(p == t.end()))
	_bad_Df_Dt(Df, Dt);

    return p->second;
}


inline upsampling_kernel_t get_kernel(int Df, int Dt)
{
    if (Df > 8) {
	if (_unlikely(Df % 8 != 0))
	    _bad_Df_Dt(Df,Dt);
	if (Dt <= 8)
	    return _get_kernel(global_kernel_table_Dt, Dt, Df, Dt);
	if (_unlikely(Dt % 8 != 0))
	    _bad_Df_Dt(Df,Dt);
	return kernel_upsample_weights;
    }

    if (Dt > 8) {
	if (_unlikely(Dt % 8 != 0))
	    _bad_Df_Dt(Df,Dt);
	return _get_kernel(global_kernel_table_Df, Df, Df, Dt);
    }
		
    array<int,2> key{{ Df, Dt }};
    return _get_kernel(global_kernel_table_Df_Dt, key, Df, Dt);
}


template<int Df, int DtMax, typename enable_if<(DtMax==0),int>::type=0>
inline void _populate_global_kernel_table_1d()
{
    global_kernel_table_Df[Df] = kernel_upsample_weights_Df<Df>;
}

template<int Df, int DtMax, typename enable_if<(DtMax>0),int>::type=0>
inline void _populate_global_kernel_table_1d()
{
    _populate_global_kernel_table_1d<Df,(DtMax/2)> ();

    array<int,2> key{{Df,DtMax}};
    global_kernel_table_Df_Dt[key] = kernel_upsample_weights_Df_Dt<Df,DtMax>;
}


template<int DfMax, int DtMax, typename enable_if<(DfMax==0 && DtMax==0),int>::type=0>
inline void _populate_global_kernel_table_2d()
{ }

template<int DfMax, int DtMax, typename enable_if<(DfMax==0 && DtMax>0),int>::type=0>
inline void _populate_global_kernel_table_2d()
{
    _populate_global_kernel_table_2d<0,(DtMax/2)> ();
    global_kernel_table_Dt[DtMax] = kernel_upsample_weights_Dt<DtMax>;
}

template<int DfMax, int DtMax, typename enable_if<(DfMax>0),int>::type=0>
inline void _populate_global_kernel_table_2d()
{
    _populate_global_kernel_table_2d<(DfMax/2),DtMax> ();
    _populate_global_kernel_table_1d<DfMax,DtMax> ();    
}


static void populate_global_kernel_table()
{
    _populate_global_kernel_table_2d<8,8> ();
}


namespace {
    struct X {
	X() { populate_global_kernel_table(); }
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
