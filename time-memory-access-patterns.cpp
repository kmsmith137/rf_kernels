#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"

#include "simd_helpers/simd_float32.hpp"
#include "simd_helpers/simd_ntuple.hpp"

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


template<unsigned int N, typename enable_if<(N==0),int>::type = 0>
inline void read_ntuple(simd_ntuple<float,S,N> &dst, const float *src, int stride) { }

template<unsigned int N, typename enable_if<(N>0),int>::type = 0>
inline void read_ntuple(simd_ntuple<float,S,N> &dst, const float *src, int stride)
{
    read_ntuple(dst.v, src, stride);
    dst.x = simd_load<float,S> (src + (N-1)*stride);
}


// -------------------------------------------------------------------------------------------------


template<unsigned int N>
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


template<unsigned int N>
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


template<unsigned int N>
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


struct memory_access_timing_thread : public kernel_timing_thread {
    memory_access_timing_thread(const shared_ptr<timing_thread_pool> &pool_, const kernel_timing_params &params_) :
	kernel_timing_thread(pool_, params_)
    { }

    virtual void thread_body() override 
    {
	rf_assert(nfreq % 16 == 0);
	rf_assert(nt_chunk % 64 == 0);

	this->allocate();

	// read_rows<N>
	
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_rows<1> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2("read_rows<1>");
	
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_rows<2> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2("read_rows<2>");
	
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_rows<4> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2("read_rows<4>");
	
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_rows<8> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2("read_rows<8>");

	// read_rows_2arr<N>
	
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_rows_2arr<1> (nfreq, nt_chunk, stride, intensity, weights);
	this->stop_timer2("read_rows_2arr<1>");

	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_rows_2arr<2> (nfreq, nt_chunk, stride, intensity, weights);
	this->stop_timer2("read_rows_2arr<2>");

	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_rows_2arr<4> (nfreq, nt_chunk, stride, intensity, weights);
	this->stop_timer2("read_rows_2arr<4>");

	// read_cols<N>
	
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_cols<1> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2("read_cols<1>");
	
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_cols<2> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2("read_cols<2>");
	
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_cols<4> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2("read_cols<4>");
	
	this->start_timer();
	for (int iter = 0; iter < niter; iter++)
	    read_cols<8> (nfreq, nt_chunk, stride, intensity);
	this->stop_timer2("read_cols<8>");
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