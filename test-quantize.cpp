#include "rf_kernels/quantize.hpp"
#include "rf_kernels/unit_testing.hpp"

using namespace std;
using namespace rf_kernels;


static void reference_quantize(int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride, int nbits)
{
    assert(nbits == 1);

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int i = 0; i < nt/8; i++) {
	    uint8_t iout = 0;
	    for (int j = 0; j < 8; j++)
		if (in[ifreq*istride + 8*i + j] > 0.0f)
		    iout |= (1 << j);

	    out[ifreq*ostride + i] = iout;
	}
    }
}


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
    reference_quantize(nfreq, nt, dst2.get(), ostride, src.get(), istride, nbits);

    for (int ifreq = 0; ifreq < nfreq; ifreq++)
	for (int i = 0; i < (nt*nbits)/8; i++)
	    assert(dst1[ifreq*ostride+i] == dst2[ifreq*ostride+i]);
}


static void test_quantize(std::mt19937 &rng)
{
#ifdef __AVX__
    constexpr int simd_nbits = 256;
#else
    constexpr int simd_nbits = 128;
#endif

    int nbits = 1;
    int nfreq = randint(rng, 1, 10);
    int nt = (simd_nbits / nbits) * randint(rng, 1, 10);
    int istride = nt + randint(rng, 0, 10);
    int ostride = (nt * nbits) / 8 + 4 * randint(rng, 0, 10);

    test_quantize(rng, nfreq, nt, ostride, istride, nbits);
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    cout << "test_quantize: start" << endl;

    for (int iter = 0; iter < 1000; iter++)
	test_quantize(rng);

    cout << "test_quantize: pass" << endl;
    return 0;
}

