#include <array>
#include <utility>
#include <memory>
#include <algorithm>
#include <functional>
#include <unordered_map>

#include "rf_kernels/internals.hpp"
#include "rf_kernels/upsample.hpp"
#include "rf_kernels/upsample_internals.hpp"

using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


constexpr int upsample_weights_max_Df = 8;
constexpr int upsample_weights_max_Dt = 8;

// Usage: kernel(nfreq_in, nt_in, dst, dstride, src, sstride, w_cutoff, Df, Dt)
using upsampling_kernel_t = void (*)(int, int, float *, int, const float *, int, float, int, int);


// -------------------------------------------------------------------------------------------------
//
// global kernel table


static unordered_map<array<int,2>, upsampling_kernel_t> global_kernel_table;

inline upsampling_kernel_t get_kernel(int Df, int Dt)
{
    array<int,2> key{{ Df, Dt }};
    auto p = global_kernel_table.find(key);

    if (_unlikely(p == global_kernel_table.end())) {
	stringstream ss;
	ss << "rf_kernels::weighted_upsampler: (Df,Dt)=(" << Df << "," << Dt << ") is not supported";
	throw runtime_error(ss.str());
    }

    return p->second;
}

template<int Df, int DtMax, typename enable_if<(DtMax==0),int>::type=0>
inline void _populate_global_kernel_table_1d()
{ }

template<int Df, int DtMax, typename enable_if<(DtMax>0),int>::type=0>
inline void _populate_global_kernel_table_1d()
{
    _populate_global_kernel_table_1d<Df,(DtMax/2)> ();

    array<int,2> key{{Df,DtMax}};
    global_kernel_table[key] = kernel_update_weights_Df_Dt<Df,DtMax>;
}


template<int DfMax, int DtMax, typename enable_if<(DfMax==0),int>::type=0>
inline void _populate_global_kernel_table_2d()
{ }

template<int DfMax, int DtMax, typename enable_if<(DfMax>0),int>::type=0>
inline void _populate_global_kernel_table_2d()
{
    _populate_global_kernel_table_2d<(DfMax/2),DtMax> ();
    _populate_global_kernel_table_1d<DfMax,DtMax> ();    
}


static void populate_global_kernel_table()
{
    _populate_global_kernel_table_2d<upsample_weights_max_Df, upsample_weights_max_Dt> ();
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
