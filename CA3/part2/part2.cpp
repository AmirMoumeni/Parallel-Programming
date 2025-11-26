
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <pmmintrin.h>   // SSE3
#include <omp.h>
#include <complex>



void julia_simd(
    int width, int height,
    float cr, float ci,
    int maxIter,
    int n,
    std::vector<unsigned char>& output)
{
    output.resize(width * height * 3);

    const __m128 c_re = _mm_set1_ps(cr);
    const __m128 c_im = _mm_set1_ps(ci);
    const __m128 four  = _mm_set1_ps(4.0f);
    const float scaleX = 4.0f / width;
    const float scaleY = 4.0f / height;

    for (int y = 0; y < height; y++) {
        __m128 zy_base = _mm_set1_ps((y - height / 2.0f) * scaleY);

        for (int x = 0; x < width; x += 4) {
            __m128 zx = _mm_set_ps(
                (x + 3 - width/2.0f) * scaleX,
                (x + 2 - width/2.0f) * scaleX,
                (x + 1 - width/2.0f) * scaleX,
                (x + 0 - width/2.0f) * scaleX
            );

            __m128 zy = zy_base;
            __m128 iter = _mm_setzero_ps();

            for (int i = 0; i < maxIter; i++) {
                // Save original z
                __m128 zr = zx;
                __m128 zi = zy;

                // Compute z^n
                __m128 zr_acc = zr;
                __m128 zi_acc = zi;

                for (int p = 1; p < n; p++) {
                    __m128 tmp_zr = _mm_sub_ps(_mm_mul_ps(zr_acc, zr), _mm_mul_ps(zi_acc, zi));
                    __m128 tmp_zi = _mm_add_ps(_mm_mul_ps(zr_acc, zi), _mm_mul_ps(zi_acc, zr));
                    zr_acc = tmp_zr;
                    zi_acc = tmp_zi;
                }

                // Add c
                zr_acc = _mm_add_ps(zr_acc, c_re);
                zi_acc = _mm_add_ps(zi_acc, c_im);

                // Compute magnitude squared
                __m128 mag2 = _mm_add_ps(_mm_mul_ps(zr_acc, zr_acc), _mm_mul_ps(zi_acc, zi_acc));

                // Mask: |z|^2 < 4
                __m128 mask = _mm_cmplt_ps(mag2, four);
                int m = _mm_movemask_ps(mask);
                if (m == 0) break;

                // Increment iter for active pixels
                iter = _mm_add_ps(iter, _mm_and_ps(mask, _mm_set1_ps(1.0f)));

                // Update zx, zy for next iteration
                zx = zr_acc;
                zy = zi_acc;
            }

            // Store results
            float iters[4];
            _mm_storeu_ps(iters, iter);
            for (int i = 0; i < 4; i++) {
                int color = std::min(255, int(255.0f * iters[i] / maxIter));
                int idx = ((y * width) + (x + i)) * 3;
                output[idx + 0] = color;
                output[idx + 1] = color;
                output[idx + 2] = color;
            }
        }
    }
}
void julia_serial(
    int width, int height,
    float cr, float ci,
    int maxIter, int n,
    std::vector<unsigned char>& output)
{
    output.resize(width * height * 3);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            float zx = (x - width / 2.0) * 4.0 / width;
            float zy = (y - height / 2.0) * 4.0 / height;

            std::complex<double> z(zx, zy);
            std::complex<double> c(cr, ci);

            int iter = 0;
            while (iter < maxIter && std::abs(z) < 2.0) {
                z = std::pow(z, n) + c;
                iter++;
            }

            // grayscale color
            int color = (int)(255.0 * iter / maxIter);

            int idx = (y * width + x) * 3;
            output[idx + 0] = color;
            output[idx + 1] = color;
            output[idx + 2] = color;
        }
    }
}

void julia_openmp(
    int width, int height,
    float cr, float ci,
    int maxIter, int n,
    std::vector<unsigned char>& output)
{
    output.resize(width * height * 3);

    // Parallelize outer loop over rows
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            double zx = (x - width / 2.0) * 4.0 / width;
            double zy = (y - height / 2.0) * 4.0 / height;

            std::complex<double> z(zx, zy);
            std::complex<double> c(cr, ci);

            int iter = 0;
            while (iter < maxIter && std::abs(z) < 2.0) {
                z = std::pow(z, n) + c;
                ++iter;
            }

            int color = static_cast<int>(255.0 * iter / maxIter);

            int idx = (y * width + x) * 3;
            output[idx + 0] = color;
            output[idx + 1] = color;
            output[idx + 2] = color;
        }
    }
}



void julia_openmp_simd(
    int width, int height,
    float cr, float ci,
    int maxIter,
    int n,
    std::vector<unsigned char>& output)
{
    output.resize(width * height * 3);

    const __m128 c_re = _mm_set1_ps(cr);
    const __m128 c_im = _mm_set1_ps(ci);
    const __m128 four  = _mm_set1_ps(4.0f);
    const float scaleX = 4.0f / width;
    const float scaleY = 4.0f / height;

    // Parallelize over rows
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        __m128 zy_base = _mm_set1_ps((y - height / 2.0f) * scaleY);

        for (int x = 0; x < width; x += 4) {
            __m128 zx = _mm_set_ps(
                (x + 3 - width/2.0f) * scaleX,
                (x + 2 - width/2.0f) * scaleX,
                (x + 1 - width/2.0f) * scaleX,
                (x + 0 - width/2.0f) * scaleX
            );

            __m128 zy = zy_base;
            __m128 iter = _mm_setzero_ps();

            for (int i = 0; i < maxIter; i++) {
                __m128 zr = zx;
                __m128 zi = zy;

                // Compute z^n
                __m128 zr_acc = zr;
                __m128 zi_acc = zi;
                for (int p = 1; p < n; p++) {
                    __m128 tmp_zr = _mm_sub_ps(_mm_mul_ps(zr_acc, zr), _mm_mul_ps(zi_acc, zi));
                    __m128 tmp_zi = _mm_add_ps(_mm_mul_ps(zr_acc, zi), _mm_mul_ps(zi_acc, zr));
                    zr_acc = tmp_zr;
                    zi_acc = tmp_zi;
                }

                // Add c
                zr_acc = _mm_add_ps(zr_acc, c_re);
                zi_acc = _mm_add_ps(zi_acc, c_im);

                // Magnitude squared
                __m128 mag2 = _mm_add_ps(_mm_mul_ps(zr_acc, zr_acc), _mm_mul_ps(zi_acc, zi_acc));

                __m128 mask = _mm_cmplt_ps(mag2, four);
                int m = _mm_movemask_ps(mask);
                if (m == 0) break;

                iter = _mm_add_ps(iter, _mm_and_ps(mask, _mm_set1_ps(1.0f)));

                zx = zr_acc;
                zy = zi_acc;
            }

            float iters[4];
            _mm_storeu_ps(iters, iter);
            for (int i = 0; i < 4; i++) {
                int color = std::min(255, int(255.0f * iters[i] / maxIter));
                int idx = ((y * width) + (x + i)) * 3;
                output[idx + 0] = color;
                output[idx + 1] = color;
                output[idx + 2] = color;
            }
        }
    }
}
int main() {
    const int width = 1000;
    const int height = 1000;
    const int maxIter = 1000;
    const int n = 2;

    const float cr = 0.355;
    const float ci = 0.355;

    std::vector<unsigned char> img_serial;
    std::vector<unsigned char> img_omp;
    std::vector<unsigned char> img_simd;
    std::vector<unsigned char> img_omp_simd;

   //SERIAL
    auto t_serial_start = std::chrono::high_resolution_clock::now();
    julia_serial(width,height,cr,ci,maxIter,n,img_serial);
    auto t_serial_end = std::chrono::high_resolution_clock::now();

    //SIMD
    auto t_simd_start = std::chrono::high_resolution_clock::now();
    julia_simd(width,height,cr,ci,maxIter,n,img_simd);
    auto t_simd_end = std::chrono::high_resolution_clock::now();
    //omp
    auto t_omp_start = std::chrono::high_resolution_clock::now();
    julia_openmp(width,height,cr,ci,maxIter,n,img_omp);
    auto t_omp_end = std::chrono::high_resolution_clock::now();


    //omp+simd
    auto t_omp_simd_start = std::chrono::high_resolution_clock::now();
    julia_openmp_simd(width,height,cr,ci,maxIter,n,img_omp_simd);
    auto t_omp_simd_end = std::chrono::high_resolution_clock::now();

    
    double t_serial = std::chrono::duration<double>(t_serial_end - t_serial_start).count();
    double t_simd = std::chrono::duration<double>(t_simd_end - t_simd_start).count();
    double t_omp = std::chrono::duration<double>(t_omp_end - t_omp_start).count();
    double t_omp_simd = std::chrono::duration<double>(t_omp_simd_end - t_omp_simd_start).count();


    std::cout << "Serial time   = " << t_serial << " s\n";
    std::cout << "simd time   = " << t_simd << " s\n";
    std::cout << "omp time   = " << t_omp << " s\n";
    std::cout << "omp+simd time   = " << t_omp_simd << " s\n";
    stbi_write_png("julia_serial.png", width, height, 3, img_serial.data(), width * 3);
    stbi_write_png("julia_simd.png", width, height, 3, img_simd.data(), width * 3);
    stbi_write_png("julia_omp.png", width, height, 3, img_omp.data(), width * 3);
    stbi_write_png("julia_omp_simd.png", width, height, 3, img_omp_simd.data(), width * 3);

        std::vector<std::pair<std::string, double>> times = {
        {"Serial", t_serial},
        {"SIMD", t_simd},
        {"OpenMP", t_omp},
        {"OpenMP+SIMD", t_omp_simd}
    };

    // Compute all pairwise speedups
    for (size_t i = 0; i < times.size(); i++) {
        for (size_t j = 0; j < times.size(); j++) {
            if (i == j) continue;
            double speedup = times[j].second / times[i].second;
            std::cout << times[i].first << " / " << times[j].first
                      << " = " << speedup << "x\n";
        }
    }

    return 0;
}
 
