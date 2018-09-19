#include <simd_helpers/core.hpp>   // simd_size()
#include "rf_kernels/quantize.hpp"
#include "rf_kernels/unit_testing.hpp"

using namespace std;
using namespace rf_kernels;


// Assumes ostride, istride positive.
static void test_quantize(std::mt19937 &rng, int nfreq, int nt, int ostride, int istride, int nbits)
{
    unique_ptr<uint8_t[]> dst1(new uint8_t[nfreq * ostride]);
    unique_ptr<uint8_t[]> dst2(new uint8_t[nfreq * ostride]);
    unique_ptr<float[]> src(new float[nfreq * istride]);

    for (int ifreq = 0; ifreq < nfreq; ifreq++)
	for (int it = 0; it < nt; it++)
	    src[ifreq*istride + it] = randint(rng,0,4) ? uniform_rand(rng,-1.0,1.0) : 0.0;

    quantizer q(nbits);
    q.quantize(nfreq, nt, dst1.get(), ostride, src.get(), istride);
    q.slow_reference_quantize(nfreq, nt, dst2.get(), ostride, src.get(), istride);

    for (int ifreq = 0; ifreq < nfreq; ifreq++)
	for (int i = 0; i < (nt*nbits)/8; i++)
	    assert(dst1[ifreq*ostride+i] == dst2[ifreq*ostride+i]);
}


static void test_quantize(std::mt19937 &rng)
{
    constexpr int simd_nbits = simd_helpers::simd_size<int>() * sizeof(int) * 8;

    int nbits = 1;
    int nfreq = randint(rng, 1, 10);
    int nt = (simd_nbits / nbits) * randint(rng, 1, 10);
    int istride = nt + randint(rng, 0, 10);
    int ostride = (nt * nbits) / 8 + 4 * randint(rng, 0, 10);

    test_quantize(rng, nfreq, nt, ostride, istride, nbits);
}


// -------------------------------------------------------------------------------------------------
//
// test_apply_bitmask


static void reference_apply_bitmask(int nfreq, int nt, float *out, int ostride, const uint8_t *in, int istride)
{
    assert(nt % 8 == 0);

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int i = 0; i < nt/8; i++) {
	    int b = in[ifreq*istride + i];
	    for (int j = 0; j < 8; j++)
		if ((b & (1 << j)) == 0)
		    out[ifreq*ostride + 8*i+j] = 0.0;
	}
    }
}


// Assumes ostride, istride positive.
static void test_apply_bitmask(std::mt19937 &rng, int nfreq, int nt, int ostride, int istride)
{
    unique_ptr<uint8_t[]> src(new uint8_t[nfreq * istride]);
    unique_ptr<float[]> dst1(new float[nfreq * ostride]);
    unique_ptr<float[]> dst2(new float[nfreq * ostride]);

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int i = 0; i < nt/8; i++)
	    src[ifreq*istride + i] = randint(rng,0,256);
	for (int it = 0; it < nt; it++)
	    dst1[ifreq*ostride + it] = dst2[ifreq*ostride + it] = uniform_rand(rng);
    }
    
    reference_apply_bitmask(nfreq, nt, dst1.get(), ostride, src.get(), istride);

    dequantizer dq(1);
    dq.apply_bitmask(nfreq, nt, dst2.get(), ostride, src.get(), istride);

    for (int ifreq = 0; ifreq < nfreq; ifreq++)
	for (int it = 0; it < nt; it++)
	    assert(dst1[ifreq*ostride+it] == dst2[ifreq*ostride+it]);
}


static void test_apply_bitmask(std::mt19937 &rng)
{
#ifdef __AVX__
    constexpr int simd_nbits = 256;
#else
    constexpr int simd_nbits = 128;
#endif

    int nfreq = randint(rng, 1, 10);
    int nt = simd_nbits * randint(rng, 1, 10);
    int istride = (nt/8) + 4 * randint(rng, 0, 10);
    int ostride = nt + randint(rng, 0, 10);

    test_apply_bitmask(rng, nfreq, nt, ostride, istride);
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    cout << "test_quantize: start" << endl;

    for (int iter = 0; iter < 1000; iter++) {
	test_quantize(rng);
	test_apply_bitmask(rng);
    }

    cout << "test_quantize: pass" << endl;
    return 0;
}

