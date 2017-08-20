#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/spline_detrender.hpp"

using namespace std;
using namespace rf_kernels;


// -------------------------------------------------------------------------------------------------
//
// Helpers


inline float xsqrt(float x)
{
    rf_assert(x >= 0.0);
    return sqrt(x);
}


inline float eval_cubic(float lval, float lderiv, float rval, float rderiv, float x)
{
    // P(x) = x^2 Q(x) + (1-x)^2 R(x)
    //
    // Q(1) = rval
    // Q'(1) = rderiv - 2*rval
    // R(0) = lval
    // R'(0) = lderiv + 2*lval

    float q = (rderiv-2*rval)*(x-1) + rval;
    float r = (lderiv+2*lval)*x + lval;
    return x*x*q + (1-x)*(1-x)*r;
}


void test_eval_cubic(std::mt19937 &rng)
{
    for (int iter = 0; iter < 10; iter++) {
	float x = uniform_rand(rng, 0.0, 1.0);

	float eps1 = 1.0 - eval_cubic(1,0,1,0,x);
	float eps2 = x - eval_cubic(0,1,1,1,x);
	float eps3 = x*x - eval_cubic(0,0,1,2,x);
	float eps4 = x*x*x - eval_cubic(0,0,1,3,x);

	rf_assert(abs(eps1) < 1.0e-6);
	rf_assert(abs(eps2) < 1.0e-6);
	rf_assert(abs(eps3) < 1.0e-6);
	rf_assert(abs(eps4) < 1.0e-6);
    }

    cout << "test_eval_cubic: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


struct vec2 {
    float v0 = 0.0;
    float v1 = 0.0;

    vec2() { }
    vec2(float v0_, float v1_) : v0(v0_), v1(v1_) { }

    vec2 &operator+=(const vec2 &v) { v0 += v.v0; v1 += v.v1; return *this; }
    vec2 operator-(const vec2 &v) { return vec2(v0-v.v0, v1-v.v1); }
};


struct mat2 {
    float a00 = 0.0;
    float a01 = 0.0;
    float a10 = 0.0;
    float a11 = 0.0;

    mat2() { }
    mat2(float a00_, float a01_, float a10_, float a11_) : a00(a00_), a01(a01_), a10(a10_), a11(a11_) { }

    mat2 &operator+=(const mat2 &m)  { a00 += m.a00; a01 += m.a01; a10 += m.a10; a11 += m.a11; return *this; }
    mat2 &operator*=(float t)        { a00 *= t; a01 *= t; a10 *= t; a11 *= t; return *this; }
    mat2 operator-(const mat2 &m)    { return mat2(a00-m.a00, a01-m.a01, a10-m.a10, a11-m.a11); }

    inline mat2 transpose() const { return mat2(a00,a10,a01,a11); }

    inline vec2 operator*(const vec2 &v) const { return vec2(a00*v.v0 + a01*v.v1, a10*v.v0 + a11*v.v1); }

    inline mat2 operator*(const mat2 &m) const
    {
	return mat2(a00 * m.a00 + a01 * m.a10,   // (0,0)
		    a00 * m.a01 + a01 * m.a11,   // (0,1)
		    a10 * m.a00 + a11 * m.a10,   // (1,0)
		    a10 * m.a01 + a11 * m.a11);  // (1,1)
    }
};


// Assumes 'a' is a symmetric matrix!
static void cholesky(mat2 &l, mat2 &linv, const mat2 &a)
{
    l.a00 = xsqrt(a.a00);
    l.a01 = 0.0;
    l.a10 = a.a01 / l.a00;
    l.a11 = xsqrt(a.a11 - l.a10*l.a10);

    // These checks should apply if all weights are positive.
    rf_assert(a.a00 > 0.0);
    rf_assert(a.a11 > 0.0);
    rf_assert(l.a10*l.a10 <= 0.999 * a.a11);
    
    linv.a00 = 1.0 / l.a00;
    linv.a01 = 0.0;
    linv.a10 = -l.a10 / (l.a00 * l.a11);
    linv.a11 = 1.0 / l.a11;
}


// Assumes 'a' is a symmetric matrix!
static vec2 solve(const mat2 &a, const vec2 &v)
{
    mat2 l, linv;
    cholesky(l, linv, a);
    
    return linv.transpose() * (linv * v);
}


inline ostream &operator<<(ostream &os, const vec2 &v) 
{ 
    os << "vec2(" << v.v0 << ", " << v.v1 << ")";
    return os;
}

inline ostream &operator<<(ostream &os, const mat2 &m)
{
    os << "mat2(" << m.a00 << ", " << m.a01 << ", " << m.a10 << ", " << m.a11 << ")";
    return os;
}


// -------------------------------------------------------------------------------------------------
//
// refsd_*: a reference implementation of spline detrending (1D).


struct refsd_params {
    const int nx;
    const int nbins;

    vector<int> bin_delim;
    vector<float> poly_vals;
    
    refsd_params(int nx, int nbins);

    // 'out' has length nx, 'coeffs' has length (2*nbins+2).
    void eval_model(float *out, const float *coeffs);

    // Like eval_model(), except that instead of using precomputed poly_vals,
    // we evaluate polynomials "from scratch" using eval_cubic().
    void eval_model_from_scratch(float *out, const float *coeffs);

    // 'ninv' has length 16, 'ninvx' has length 4, intensity+weights have length nx.
    void analyze_bin(float *ninv, float *ninvx, const float *intensity, const float *weights, int b);

    void fit_model(float *coeffs, const float *intensity, const float *weights);
};


refsd_params::refsd_params(int nx_, int nbins_) :
    nx(nx_), 
    nbins(nbins_),
    bin_delim(nbins_+1, 0),
    poly_vals(4*nx_, 0.0)
{
    _spline_detrender_init(&bin_delim[0], &poly_vals[0], nx, nbins);    
}


void refsd_params::eval_model(float *out, const float *coeffs)
{
    memset(out, 0, nx * sizeof(*out));

    for (int b = 0; b < nbins; b++)
	for (int i = bin_delim[b]; i < bin_delim[b+1]; i++)
	    for (int p = 0; p < 4; p++)
		out[i] += coeffs[2*b+p] * poly_vals[4*i+p];
}


void refsd_params::eval_model_from_scratch(float *out, const float *coeffs)
{
    for (int i = 0; i < nx; i++) {
	float x = (i+0.5) / float(nx) * float(nbins);   // note: 0 < x < nbins
	int b = int(x);
	out[i] = eval_cubic(coeffs[2*b], coeffs[2*b+1], coeffs[2*b+2], coeffs[2*b+3], x-b);
    }
}


void refsd_params::analyze_bin(float ninv[16], float ninvx[4], const float *intensity, const float *weights, int b)
{
    memset(ninv, 0, 16 * sizeof(ninv[0]));
    memset(ninvx, 0, 4 * sizeof(ninvx[0]));

    for (int i = bin_delim[b]; i < bin_delim[b+1]; i++) {
	for (int p = 0; p < 4; p++)
	    for (int q = 0; q < 4; q++)
		ninv[4*p+q] += weights[i] * poly_vals[4*i+p] * poly_vals[4*i+q];
	
	for (int p = 0; p < 4; p++)
	    ninvx[p] += weights[i] * intensity[i] * poly_vals[4*i+p];
    }
}


void refsd_params::fit_model(float *coeffs, const float *intensity, const float *weights)
{
    mat2 a_prev;
    vec2 v_prev;
    
    for (int b0 = 0; b0 < nbins; b0++) {
	float ninv[16];
	float ninvx[4];
	analyze_bin(ninv, ninvx, intensity, weights, b0);

	mat2 a(ninv[0], ninv[1], ninv[4], ninv[5]);
	mat2 b(ninv[2], ninv[3], ninv[6], ninv[7]);
	mat2 c(ninv[10], ninv[11], ninv[14], ninv[15]);
	vec2 v(ninvx[0], ninvx[1]);
	vec2 w(ninvx[2], ninvx[3]);

	a += a_prev;
	v += v_prev;

	mat2 l, linv;
	cholesky(l, linv, a);

	mat2 linvb = linv * b;

	// B^T A^{-1} B = B^T L^{-T} L^{-1} B
	// B^T A^{-1} v = B^T L^{-T} L^{-1} v
	a_prev = c - linvb.transpose() * linvb;
	v_prev = w - linvb.transpose() * (linv * v);
	// cout << "A = " << a_prev << ", v = " << v_prev << endl;
    }

    vec2 c_prev = solve(a_prev, v_prev);
    coeffs[2*nbins] = c_prev.v0;
    coeffs[2*nbins+1] = c_prev.v1;

    for (int b0 = nbins-1; b0 >= 0; b0--) {
	float ninv[16];
	float ninvx[4];
	analyze_bin(ninv, ninvx, intensity, weights, b0);
	
	mat2 a(ninv[0], ninv[1], ninv[4], ninv[5]);
	mat2 b(ninv[2], ninv[3], ninv[6], ninv[7]);
	vec2 v(ninvx[0], ninvx[1]);

	c_prev = solve(a, v - b * c_prev);
	// c_prev = solve(a, b * c_prev);
	coeffs[2*b0] = c_prev.v0;
	coeffs[2*b0+1] = c_prev.v1;
	// cout << "A = " << a << ", b = " << b << ", c = " << c_prev << endl;
    }
}


// -------------------------------------------------------------------------------------------------


static void test_reference_spline_detrender(std::mt19937 &rng, int nx, int nbins)
{
    refsd_params params(nx, nbins);
    
    vector<float> intensity(nx, 0.0);
    vector<float> intensity2(nx, 0.0);
    // vector<float> weights = uniform_randvec(rng, nx, 0.1, 1.0);
    vector<float> weights(nx, 1.0);

    vector<float> in_coeffs = uniform_randvec(rng, 2*nbins+2, -1.0, 1.0);
    params.eval_model(&intensity[0], &in_coeffs[0]);
    params.eval_model_from_scratch(&intensity2[0], &in_coeffs[0]);

    cout << "eval_model comparison: " << nx << " " << nbins << " " << maxdiff(intensity,intensity2) << endl;
    
    vector<float> out_coeffs(2*nbins+2, 0.0);
    params.fit_model(&out_coeffs[0], &intensity[0], &weights[0]);

    cout << "fit_model comparison: " << nx << " " << nbins << " " << maxdiff(in_coeffs,out_coeffs) << endl;
}


static void test_reference_spline_detrender(std::mt19937 &rng)
{
    for (int iter = 0; iter < 5; iter++) {
	// int nx = randint(rng, 32, 1000);
	// int nbins = randint(rng, 1, min(nx/32+1,21));
	int nx = 1600;
	int nbins = 40;
	test_reference_spline_detrender(rng, nx, nbins);
    }

    cout << "test_reference_spline_detrender: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_eval_cubic(rng);
    test_reference_spline_detrender(rng);

    return 0;
}
