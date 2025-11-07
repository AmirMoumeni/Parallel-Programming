#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>  
#include <pmmintrin.h>  

#define N 10000000  

void generate_random_floats(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 100.0f; 
    }
}
float variance_serial(float *arr, int n) {
    float sum = 0.0f;
    float sum_sq = 0.0f;

    
    for (int i = 0; i < n; i++) {
        float val = arr[i];
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / n;
    float var = (sum_sq / n) - (mean * mean);  
    return var;
}


float variance_sse3(float *arr, int n) {
    __m128 sum_vec = _mm_setzero_ps();    
    __m128 sumsq_vec = _mm_setzero_ps();   

    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __m128 v = _mm_loadu_ps(&arr[i]);            
        sum_vec = _mm_add_ps(sum_vec, v);            
        __m128 v_sq = _mm_mul_ps(v, v);              
        sumsq_vec = _mm_add_ps(sumsq_vec, v_sq);     
    }

   
    sum_vec   = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec   = _mm_hadd_ps(sum_vec, sum_vec);
    sumsq_vec = _mm_hadd_ps(sumsq_vec, sumsq_vec);
    sumsq_vec = _mm_hadd_ps(sumsq_vec, sumsq_vec);

    float sum   = _mm_cvtss_f32(sum_vec); 
    float sumsq = _mm_cvtss_f32(sumsq_vec);

    for (; i < n; i++) {
        float val = arr[i];
        sum += val;
        sumsq += val * val;
    }

    float mean = sum / n;
    float var  = (sumsq / n) - (mean * mean);  
    return var;
}



int main() {
    srand(time(NULL));
    float *data = (float*)malloc(N * sizeof(float));

    generate_random_floats(data, N);

    clock_t start, end;

    start = clock();
    float var_serial = variance_serial(data, N);
    end = clock();
    double time_serial = (double)(end - start) / CLOCKS_PER_SEC;

    start = clock();
    float var_sse3 = variance_sse3(data, N);
    end = clock();
    double time_sse3 = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Serial Variance: %f, Time: %f s\n", var_serial, time_serial);
    printf("SSE3 Variance:   %f, Time: %f s\n", var_sse3, time_sse3);
    printf("Speedup: %fx\n", time_serial / time_sse3);
    
    free(data);
    return 0;
}
