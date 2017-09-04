#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"

#include <simd_helpers/simd_float32.hpp>
#include <simd_helpers/simd_ntuple.hpp>

#ifdef __AVX__
constexpr int S = 8;
#else
constexpr int S = 4;
#endif

using namespace std;
using namespace rf_kernels;
using namespace simd_helpers;


// -------------------------------------------------------------------------------------------------


inline void throw_away(const simd_t<float,S> &x)
{
    float t = x.sum();
    if (t > 0.0)
	usleep(1);
}


template<int N, typename enable_if<(N==0),int>::type = 0>
inline void read_ntuple(simd_ntuple<float,S,N> &dst, const float *src, int stride) { }

template<int N, typename enable_if<(N>0),int>::type = 0>
inline void read_ntuple(simd_ntuple<float,S,N> &dst, const float *src, int stride)
{
    read_ntuple(dst.v, src, stride);
    dst.x = simd_load<float,S> (src + (N-1)*stride);
}


template<int N, typename enable_if<(N==0),int>::type = 0>
inline void write_ntuple(float *dst, const simd_ntuple<float,S,N> &src, int stride) { }

template<int N, typename enable_if<(N>0),int>::type = 0>
inline void write_ntuple(float *dst, const simd_ntuple<float,S,N> &src, int stride)
{
    write_ntuple(dst, src.v, stride);
    simd_store(dst + (N-1)*stride, src.x);
}


// -------------------------------------------------------------------------------------------------


template<int N>
inline void read_rows(int nfreq, int nt_chunk, int stride, const float *src)
{
    simd_t<float,S> acc(0.0);
	
    for (int ifreq = 0; ifreq < nfreq; ifreq += N) {
	for (int it = 0; it < nt_chunk; it += S) {
	    simd_ntuple<float,S,N> x;
	    read_ntuple(x, src + ifreq*stride + it, stride);
	    acc += x.vertical_sum();
	}
    }

    throw_away(acc);
}


template<int N>
inline void read_rows_2arr(int nfreq, int nt_chunk, int stride, const float *src1, const float *src2)
{
    simd_t<float,S> acc(0.0);
	
    for (int ifreq = 0; ifreq < nfreq; ifreq += N) {
	const float *row1 = src1 + ifreq*stride;
	const float *row2 = src2 + ifreq*stride;

	for (int it = 0; it < nt_chunk; it += S) {
	    simd_ntuple<float,S,N> x1;
	    simd_ntuple<float,S,N> x2;
	    read_ntuple(x1, row1 + it, stride);
	    read_ntuple(x2, row2 + it, stride);
	    acc += x1.vertical_sum() + x2.vertical_sum();
	}
    }

    throw_away(acc);
}



// -------------------------------------------------------------------------------------------------

template<int N>
inline void read_cols(int nfreq, int nt_chunk, int stride, const float *src)
{
    simd_t<float,S> acc(0.0);
	
    for (int it = 0; it < nt_chunk; it += N*S) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_ntuple<float,S,N> x;
	    read_ntuple(x, src + ifreq*stride + it, S);
	    acc += x.vertical_sum();
	}
    }

    throw_away(acc);
}


template<int N>
inline void update_cols(int nfreq, int nt_chunk, int stride, float *arr)
{
    simd_t<float,S> one(1.0);
	
    for (int it = 0; it < nt_chunk; it += N*S) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_ntuple<float,S,N> x;
	    read_ntuple(x, arr + ifreq*stride + it, S);
	    x += one;
	    write_ntuple(arr + ifreq*stride + it, x, S);
	}
    }
}


// -------------------------------------------------------------------------------------------------


void mock_fclip_in_place(int niter, int nfreq, int nt_chunk, int stride, const float *i_in, float *w_in)
{
    simd_t<float,S> acc = 0;

    for (int it = 0; it < nt_chunk; it += S) {
	for (int iter = 0; iter < niter; iter++) {
	    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
		simd_t<float,S> ival = simd_helpers::simd_load<float,S> (i_in + ifreq*stride + it);
		simd_t<float,S> wval = simd_helpers::simd_load<float,S> (w_in + ifreq*stride + it);
		acc += ival * wval;
	    }
	}
	
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_t<float,S> ival = simd_helpers::simd_load<float,S> (i_in + ifreq*stride + it);
	    simd_t<float,S> wval = simd_helpers::simd_load<float,S> (w_in + ifreq*stride + it);
	    simd_helpers::simd_store(w_in + ifreq*stride + it, wval + acc*ival);
	}
    }	    
}


void mock_fclip_tmp(int niter, int nfreq, int nt_chunk, int stride, const float *i_in, float *w_in, float *i_tmp, float *w_tmp)
{
    simd_t<float,S> acc = 0;

    for (int it = 0; it < nt_chunk; it += S) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    simd_t<float,S> ival = simd_helpers::simd_load<float,S> (i_in + ifreq*stride + it);
	    simd_t<float,S> wval = simd_helpers::simd_load<float,S> (w_in + ifreq*stride + it);
	    acc += ival * wval;

	    simd_helpers::simd_store(i_tmp + ifreq*S, ival);
	    simd_helpers::simd_store(w_tmp + ifreq*S, wval);
	}
		 
	for (int iter = 1; iter < niter; iter++) {
	    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
		simd_t<float,S> ival = simd_helpers::simd_load<float,S> (i_tmp + ifreq*S);
		simd_t<float,S> wval = simd_helpers::simd_load<float,S> (w_tmp + ifreq*S);
		acc += ival * wval;
	    }
	}
	
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
#if 1
	    simd_t<float,S> ival = simd_helpers::simd_load<float,S> (i_in + ifreq*stride + it);
	    simd_t<float,S> wval = simd_helpers::simd_load<float,S> (w_in + ifreq*stride + it);
#else
	    simd_t<float,S> ival = simd_helpers::simd_load<float,S> (i_tmp + ifreq*S);
	    simd_t<float,S> wval = simd_helpers::simd_load<float,S> (w_tmp + ifreq*S);
#endif
	    simd_helpers::simd_store(w_in + ifreq*stride + it, wval + acc*ival);
	}
    }	    
}


struct memory_access_timing_thread : public kernel_timing_thread {
    memory_access_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    float *i_tmp = nullptr;
    float *w_tmp = nullptr;

    template<int N>
    inline void time_read_rows(const char *str)
    {
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_rows<N> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2(str);
    }

    template<int N>
    inline void time_read_rows_2arr(const char *str)
    {
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_rows_2arr<N> (nfreq, nt_chunk, stride, intensity, weights);
	this->stop_timer2(str);
    }

    template<int N>
    inline void time_read_cols(const char *str)
    {
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_cols<N> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2(str);
    }

    template<int N>
    inline void time_update_cols(const char *str)
    {
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    update_cols<N> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2(str);
    }

    // This is a "mockup" of intensity_clipper(AXIS_FREQ).
    void time_fclip(int ic_iter)
    {
	if (!this->i_tmp)
	    this->i_tmp = aligned_alloc<float> (S * nfreq);
	if (!this->w_tmp)
	    this->w_tmp = aligned_alloc<float> (S * nfreq);
	    
	stringstream ss1;
	ss1 << "mock_fclip_in_place(" << ic_iter << ")";
	string s1 = ss1.str();
	const char *cp1 = s1.c_str();
	
	stringstream ss2;
	ss2 << "mock_fclip_tmp(" << ic_iter << ")";
	string s2 = ss2.str();
	const char *cp2 = s2.c_str();
	
	this->start_timer();

 	for (int iter = 0; iter < niter; iter++)
	    mock_fclip_in_place(ic_iter, nfreq, nt_chunk, stride, intensity, weights);

	this->stop_timer(cp1);

	this->start_timer();

 	for (int iter = 0; iter < niter; iter++)
	    mock_fclip_tmp(ic_iter, nfreq, nt_chunk, stride, intensity, weights, i_tmp, w_tmp);

	this->stop_timer(cp2);
		
    }

    virtual void thread_body() override 
    {
	rf_assert(nfreq % 16 == 0);
	rf_assert(nt_chunk % 64 == 0);

	this->allocate();

	for (int ic_iter = 1; ic_iter < 6; ic_iter++)
	    time_fclip(ic_iter);
	
	this->time_read_rows<1> ("read_rows<1>");
	this->time_read_rows<2> ("read_rows<2>");
	this->time_read_rows<4> ("read_rows<4>");
	this->time_read_rows<8> ("read_rows<8>");

	this->time_read_rows_2arr<1> ("read_rows_2arr<1>");	
	this->time_read_rows_2arr<2> ("read_rows_2arr<2>");	
	this->time_read_rows_2arr<4> ("read_rows_2arr<4>");	

	this->time_read_cols<1> ("read_cols<1>");
	this->time_read_cols<2> ("read_cols<2>");
	this->time_read_cols<4> ("read_cols<4>");
	this->time_read_cols<8> ("read_cols<8>");

	this->time_update_cols<1> ("update_cols<1>");
	this->time_update_cols<2> ("update_cols<2>");
	this->time_update_cols<4> ("update_cols<4>");
	this->time_update_cols<8> ("update_cols<8>");
    }
};


int main(int argc, char **argv)
{
    kernel_timing_params params("time-spline-detrender");
    params.parse_args(argc, argv);

    int nthreads = params.nthreads;
    auto pool = make_shared<timing_thread_pool> (nthreads);

    vector<std::thread> threads;
    for (int i = 0; i < nthreads; i++)
	threads.push_back(spawn_timing_thread<memory_access_timing_thread> (pool, params));
    for (int i = 0; i < nthreads; i++)
	threads[i].join();

    return 0;
}
