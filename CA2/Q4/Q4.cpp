#include <iostream>
#include <immintrin.h>
#include <vector>
#include <chrono>
#include <cmath>
using namespace std;

float relu(float x){return x>0?x:0;}

float dot_serial(const vector<float>& a,const vector<float>& b){
    float s=0;
    for(size_t i=0;i<a.size();++i)s+=a[i]*b[i];
    return s;
}

float dot_simd(const vector<float>& a,const vector<float>& b){
    size_t n=a.size();
    size_t i=0;
    __m256 sum=_mm256_setzero_ps();
    for(;i+8<=n;i+=8){
        __m256 va=_mm256_loadu_ps(&a[i]);
        __m256 vb=_mm256_loadu_ps(&b[i]);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(va,vb));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp,sum);
    float s=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for(;i<n;++i)s+=a[i]*b[i];
    return s;
}

vector<float> layer_serial(const vector<float>& in,const vector<vector<float>>& w,const vector<float>& b){
    vector<float> out(w.size());
    for(size_t i=0;i<w.size();++i)out[i]=relu(dot_serial(in,w[i])+b[i]);
    return out;
}

vector<float> layer_simd(const vector<float>& in,const vector<vector<float>>& w,const vector<float>& b){
    vector<float> out(w.size());
    for(size_t i=0;i<w.size();++i)out[i]=relu(dot_simd(in,w[i])+b[i]);
    return out;
}

int main(){
    int n_in=8,n_hid=16,n_out=8;
    vector<float> input(n_in);
    for(float&x:input)x=rand()/(float)RAND_MAX;
    vector<vector<float>> w1(n_hid,vector<float>(n_in));
    vector<vector<float>> w2(n_out,vector<float>(n_hid));
    vector<float> b1(n_hid),b2(n_out);
    for(auto&r:w1)for(float&x:r)x=rand()/(float)RAND_MAX;
    for(auto&r:w2)for(float&x:r)x=rand()/(float)RAND_MAX;
    for(float&x:b1)x=rand()/(float)RAND_MAX;
    for(float&x:b2)x=rand()/(float)RAND_MAX;

    auto t1=chrono::high_resolution_clock::now();
    auto h1s=layer_serial(input,w1,b1);
    auto outs=layer_serial(h1s,w2,b2);
    auto t2=chrono::high_resolution_clock::now();
    double ts=chrono::duration<double>(t2-t1).count();

    auto t3=chrono::high_resolution_clock::now();
    auto h1p=layer_simd(input,w1,b1);
    auto outp=layer_simd(h1p,w2,b2);
    auto t4=chrono::high_resolution_clock::now();
    double tp=chrono::duration<double>(t4-t3).count();

    cout<<"Serial time: "<<ts<<" s\n";
    cout<<"SIMD time:   "<<tp<<" s\n";
    cout<<"Speedup:     "<<ts/tp<<"x\n";
    return 0;
}
