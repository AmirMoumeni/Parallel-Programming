#include <iostream>
#include <immintrin.h>
#include <vector>
#include <chrono>
#include <omp.h>
using namespace std;

float relu(float x){ return x>0 ? x : 0; }

// ===================== Serial Dot =====================
float dot_serial(const vector<float>& a,const vector<float>& b){
    float s=0;
    for(size_t i=0;i<a.size();++i)
        s+=a[i]*b[i];
    return s;
}

// ===================== SIMD Dot =====================
float dot_simd(const vector<float>& a,const vector<float>& b){
    size_t n=a.size();
    size_t i=0;
    __m256 sum=_mm256_setzero_ps();

    for(; i+8 <= n; i+=8){
        __m256 va=_mm256_loadu_ps(&a[i]);
        __m256 vb=_mm256_loadu_ps(&b[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va,vb));
    }

    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum);
    float s = 0;
    for(int k=0;k<8;k++) s += tmp[k];

    for(; i<n; i++) s += a[i]*b[i];

    return s;
}

// ===================== Serial Layer =====================
vector<float> layer_serial(const vector<float>& in,
                           const vector<vector<float>>& w,
                           const vector<float>& b){
    vector<float> out(w.size());
    for(size_t i=0;i<w.size();++i)
        out[i] = relu(dot_serial(in,w[i]) + b[i]);
    return out;
}

// ===================== SIMD Layer =====================
vector<float> layer_simd(const vector<float>& in,
                         const vector<vector<float>>& w,
                         const vector<float>& b){
    vector<float> out(w.size());
    for(size_t i=0;i<w.size();++i)
        out[i] = relu(dot_simd(in,w[i]) + b[i]);
    return out;
}

// ===================== OpenMP (Serial dot) =====================
vector<float> layer_omp(const vector<float>& in,
                        const vector<vector<float>>& w,
                        const vector<float>& b){
    vector<float> out(w.size());
#pragma omp parallel for
    for(int i=0; i<(int)w.size(); i++)
        out[i] = relu(dot_serial(in, w[i]) + b[i]);
    return out;
}

// ===================== OpenMP + SIMD =====================
vector<float> layer_omp_simd(const vector<float>& in,
                             const vector<vector<float>>& w,
                             const vector<float>& b){
    vector<float> out(w.size());
#pragma omp parallel for
    for(int i=0; i<(int)w.size(); i++)
        out[i] = relu(dot_simd(in, w[i]) + b[i]);
    return out;
}

// ============================== MAIN ==============================
int main(){
    int n_in=2048, n_hid=2048, n_out=1024;

    vector<float> input(n_in);
    for(float&x:input)x=rand()/(float)RAND_MAX;

    vector<vector<float>> w1(n_hid, vector<float>(n_in));
    vector<vector<float>> w2(n_out, vector<float>(n_hid));
    vector<float> b1(n_hid), b2(n_out);

    for(auto&r:w1)for(float&x:r)x=rand()/(float)RAND_MAX;
    for(auto&r:w2)for(float&x:r)x=rand()/(float)RAND_MAX;
    for(float&x:b1)x=rand()/(float)RAND_MAX;
    for(float&x:b2)x=rand()/(float)RAND_MAX;


    // ===== Serial =====
    auto t1=chrono::high_resolution_clock::now();
    auto h1s = layer_serial(input,w1,b1);
    auto outs = layer_serial(h1s,w2,b2);
    auto t2=chrono::high_resolution_clock::now();
    double ts = chrono::duration<double>(t2-t1).count();

    // ===== SIMD =====
    auto t3=chrono::high_resolution_clock::now();
    auto h1v = layer_simd(input,w1,b1);
    auto outv = layer_simd(h1v,w2,b2);
    auto t4=chrono::high_resolution_clock::now();
    double tv = chrono::duration<double>(t4-t3).count();

    // ===== OpenMP =====
    auto t5=chrono::high_resolution_clock::now();
    auto h1o = layer_omp(input,w1,b1);
    auto outo = layer_omp(h1o,w2,b2);
    auto t6=chrono::high_resolution_clock::now();
    double to = chrono::duration<double>(t6-t5).count();

    // ===== OpenMP + SIMD =====
    auto t7=chrono::high_resolution_clock::now();
    auto h1ov = layer_omp_simd(input,w1,b1);
    auto outov = layer_omp_simd(h1ov,w2,b2);
    auto t8=chrono::high_resolution_clock::now();
    double tov = chrono::duration<double>(t8-t7).count();


    cout << "Serial:        " << ts  << " s\n";
    cout << "SIMD:          " << tv  << " s\n";
    cout << "OpenMP:        " << to  << " s\n";
    cout << "OMP+SIMD:      " << tov << " s\n\n";

    cout << "Speedup SIMD:       " << ts/tv  << "x\n";
    cout << "Speedup OpenMP:     " << ts/to  << "x\n";
    cout << "Speedup OMP+SIMD:   " << ts/tov << "x\n";

    return 0;
}

//g++ -O3 -march=native -mfma -ffast-math -fopenmp NN.cpp -o NN.exe

// ================= OUTPUT ==================

// Serial:        0.0025812 s
// SIMD:          0.0024059 s
// OpenMP:        0.0046775 s
// OMP+SIMD:      0.0018201 s
// 
// Speedup SIMD:       1.07286x
// Speedup OpenMP:     0.551833x
// Speedup OMP+SIMD:   1.41817x