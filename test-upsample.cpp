#include <simd_helpers/simd_debug.hpp>

#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/upsample_internals.hpp"

using namespace std;
using namespace rf_kernels;
using namespace simd_helpers;


// -------------------------------------------------------------------------------------------------
//
// test_simd_upsample<Dt> ()
//
// Eventually, this code can be generalized to a pair <T,S> and moved to simd_helpers.


// Temporary hack, which can go away when this code gets moved to simd_helpers.
template<typename T, int S> using xsimd_t = simd_helpers::simd_t<T,S>;
template<typename T, int S, int D> using xsimd_ntuple = simd_helpers::simd_ntuple<T,S,D>;


template<int Dt>
void test_simd_upsample2(std::mt19937 &rng)
{
    vector<float> src = uniform_randvec<float> (rng, 8, 0, 1000);

    xsimd_t<float,8> t = pack_simd_t<float,8> (src);
    xsimd_ntuple<float,8,Dt> u = simd_upsample<Dt> (t);
    vector<float> dst = vectorize(u);

    for (int i = 0; i < 8; i++)
	for (int j = 0; j < Dt; j++)
	    assert(src[i] == dst[i*Dt+j]);
}


void test_simd_upsample(std::mt19937 &rng)
{
    for (int iouter = 0; iouter < 1000; iouter++) {
	test_simd_upsample2<1> (rng);
	test_simd_upsample2<2> (rng);
	test_simd_upsample2<4> (rng);
	test_simd_upsample2<8> (rng);
    }

    cout << "test_simd_upsample: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_simd_upsample(rng);
    return 0;
}
