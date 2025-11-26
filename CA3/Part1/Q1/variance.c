#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <omp.h>

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

    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sumsq_vec = _mm_hadd_ps(sumsq_vec, sumsq_vec);
    sumsq_vec = _mm_hadd_ps(sumsq_vec, sumsq_vec);

    float sum = _mm_cvtss_f32(sum_vec);
    float sumsq = _mm_cvtss_f32(sumsq_vec);

    for (; i < n; i++) {
        float val = arr[i];
        sum += val;
        sumsq += val * val;
    }

    float mean = sum / n;
    float var = (sumsq / n) - (mean * mean);
    return var;
}

float variance_openmp(float *arr, int n) {
    float sum = 0.0f;
    float sum_sq = 0.0f;

    #pragma omp parallel for reduction(+:sum,sum_sq)
    for (int i = 0; i < n; i++) {
        float val = arr[i];
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / n;
    float var = (sum_sq / n) - (mean * mean);
    return var;
}

float variance_openmp_sse3(float *arr, int n) {
    float sum = 0.0f;
    float sum_sq = 0.0f;

    #pragma omp parallel reduction(+:sum,sum_sq)
    {
        __m128 sum_v = _mm_setzero_ps();
        __m128 sumsq_v = _mm_setzero_ps();

        #pragma omp for
        for (int i = 0; i <= n - 4; i += 4) {
            __m128 v = _mm_loadu_ps(&arr[i]);
            sum_v = _mm_add_ps(sum_v, v);
            __m128 v_sq = _mm_mul_ps(v, v);
            sumsq_v = _mm_add_ps(sumsq_v, v_sq);
        }

        sum_v = _mm_hadd_ps(sum_v, sum_v);
        sum_v = _mm_hadd_ps(sum_v, sum_v);
        sumsq_v = _mm_hadd_ps(sumsq_v, sumsq_v);
        sumsq_v = _mm_hadd_ps(sumsq_v, sumsq_v);

        sum += _mm_cvtss_f32(sum_v);
        sum_sq += _mm_cvtss_f32(sumsq_v);
    }

    for (int i = (n / 4) * 4; i < n; i++) {
        float val = arr[i];
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / n;
    float var = (sum_sq / n) - (mean * mean);
    return var;
}

int main() {
    srand(time(NULL));
    float *data = (float*)malloc(N * sizeof(float));
    generate_random_floats(data, N);

    double t1, t2;

    t1 = omp_get_wtime();
    float var_serial = variance_serial(data, N);
    t2 = omp_get_wtime();
    double time_serial = t2 - t1;

    t1 = omp_get_wtime();
    float var_sse3 = variance_sse3(data, N);
    t2 = omp_get_wtime();
    double time_sse3 = t2 - t1;

    t1 = omp_get_wtime();
    float var_omp = variance_openmp(data, N);
    t2 = omp_get_wtime();
    double time_omp = t2 - t1;

    t1 = omp_get_wtime();
    float var_omp_sse3 = variance_openmp_sse3(data, N);
    t2 = omp_get_wtime();
    double time_omp_sse3 = t2 - t1;

    printf("\nSerial:         %f s (var=%f)\n", time_serial, var_serial);
    printf("SSE3:           %f s (var=%f)\n", time_sse3, var_sse3);
    printf("OpenMP:         %f s (var=%f)\n", time_omp, var_omp);
    printf("OpenMP+SSE3:    %f s (var=%f)\n\n", time_omp_sse3, var_omp_sse3);

    printf("Speedup SSE3:        %fx\n", time_serial / time_sse3);
    printf("Speedup OpenMP:      %fx\n", time_serial / time_omp);
    printf("Speedup OMP+SSE3:    %fx\n", time_serial / time_omp_sse3);

    free(data);
    return 0;
}

//gcc variance.c -o variance.exe -fopenmp -msse3


// ================= OUTPUT ==================

// Serial:         0.032762 s (var=810.157959)
// SSE3:           0.017189 s (var=832.250244)
// OpenMP:         0.105041 s (var=832.308594)
// OpenMP+SSE3:    0.050150 s (var=832.603760)

// Speedup SSE3:        1.906037x
// Speedup OpenMP:      0.311901x
// Speedup OMP+SSE3:    0.653292x
