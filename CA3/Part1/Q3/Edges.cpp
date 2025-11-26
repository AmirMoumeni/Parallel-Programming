#include <iostream>
#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace cv;

// ===================== Serial =====================
Mat convolve_serial(const Mat& src, const float kernel[3][3]) {
    Mat dst = Mat::zeros(src.size(), CV_32F);

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx)
                    sum += src.at<float>(y + ky, x + kx) * kernel[ky + 1][kx + 1];

            dst.at<float>(y, x) = sum;
        }
    }
    return dst;
}

// ===================== SIMD =====================
Mat convolve_simd(const Mat& src, const float kernel[3][3]) {
    Mat dst = Mat::zeros(src.size(), CV_32F);

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 8; x += 8) {
            __m256 sum = _mm256_setzero_ps();

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    __m256 pix = _mm256_loadu_ps(&src.at<float>(y + ky, x + kx));
                    __m256 k = _mm256_set1_ps(kernel[ky + 1][kx + 1]);
                    sum = _mm256_fmadd_ps(pix, k, sum);
                }
            }

            _mm256_storeu_ps(&dst.at<float>(y, x), sum);
        }
    }
    return dst;
}

// ===================== OpenMP =====================
Mat convolve_omp(const Mat& src, const float kernel[3][3]) {
    Mat dst = Mat::zeros(src.size(), CV_32F);

#pragma omp parallel for
    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx)
                    sum += src.at<float>(y + ky, x + kx) * kernel[ky + 1][kx + 1];

            dst.at<float>(y, x) = sum;
        }
    }
    return dst;
}

// ===================== OpenMP + SIMD =====================
Mat convolve_omp_simd(const Mat& src, const float kernel[3][3]) {
    Mat dst = Mat::zeros(src.size(), CV_32F);

#pragma omp parallel for
    for (int y = 1; y < src.rows - 1; ++y) {

        for (int x = 1; x < src.cols - 8; x += 8) {

            __m256 sum = _mm256_setzero_ps();

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    __m256 pix = _mm256_loadu_ps(&src.at<float>(y + ky, x + kx));
                    __m256 k = _mm256_set1_ps(kernel[ky + 1][kx + 1]);
                    sum = _mm256_fmadd_ps(pix, k, sum);
                }
            }

            _mm256_storeu_ps(&dst.at<float>(y, x), sum);
        }
    }

    return dst;
}

// ============================== MAIN ==============================
int main() {
    Mat img = imread("base.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Image not found!\n";
        return -1;
    }

    img.convertTo(img, CV_32F);

    float gaussian_blur[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    float sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    float sobel_y[3][3] = {
        {-1, -2, -1},
        {0,  0,  0},
        {1,  2,  1}
    };

    // -------- Serial --------
    auto t1 = chrono::high_resolution_clock::now();
    Mat b_s = convolve_serial(img, gaussian_blur);
    Mat x_s = convolve_serial(b_s, sobel_x);
    Mat y_s = convolve_serial(b_s, sobel_y);
    Mat mag_s = x_s.mul(x_s) + y_s.mul(y_s);
    sqrt(mag_s, mag_s);
    auto t2 = chrono::high_resolution_clock::now();
    double ts = chrono::duration<double>(t2 - t1).count();

    // -------- SIMD --------
    t1 = chrono::high_resolution_clock::now();
    Mat b_v = convolve_simd(img, gaussian_blur);
    Mat x_v = convolve_simd(b_v, sobel_x);
    Mat y_v = convolve_simd(b_v, sobel_y);
    Mat mag_v = x_v.mul(x_v) + y_v.mul(y_v);
    sqrt(mag_v, mag_v);
    t2 = chrono::high_resolution_clock::now();
    double tv = chrono::duration<double>(t2 - t1).count();

    // -------- OpenMP --------
    t1 = chrono::high_resolution_clock::now();
    Mat b_o = convolve_omp(img, gaussian_blur);
    Mat x_o = convolve_omp(b_o, sobel_x);
    Mat y_o = convolve_omp(b_o, sobel_y);
    Mat mag_o = x_o.mul(x_o) + y_o.mul(y_o);
    sqrt(mag_o, mag_o);
    t2 = chrono::high_resolution_clock::now();
    double to = chrono::duration<double>(t2 - t1).count();

    // -------- OpenMP + SIMD --------
    t1 = chrono::high_resolution_clock::now();
    Mat b_ov = convolve_omp_simd(img, gaussian_blur);
    Mat x_ov = convolve_omp_simd(b_ov, sobel_x);
    Mat y_ov = convolve_omp_simd(b_ov, sobel_y);
    Mat mag_ov = x_ov.mul(x_ov) + y_ov.mul(y_ov);
    sqrt(mag_ov, mag_ov);
    t2 = chrono::high_resolution_clock::now();
    double tov = chrono::duration<double>(t2 - t1).count();

    // -------- Results --------
    cout << "Serial:        " << ts  << " s\n";
    cout << "SIMD:          " << tv  << " s\n";
    cout << "OpenMP:        " << to  << " s\n";
    cout << "OMP+SIMD:      " << tov << " s\n\n";

    cout << "Speedup SIMD:       " << ts / tv  << "x\n";
    cout << "Speedup OpenMP:     " << ts / to  << "x\n";
    cout << "Speedup OMP+SIMD:   " << ts / tov << "x\n";

    mag_s.convertTo(mag_s, CV_8U);
    mag_v.convertTo(mag_v, CV_8U);
    mag_o.convertTo(mag_o, CV_8U);
    mag_ov.convertTo(mag_ov, CV_8U);

    imwrite("edges_serial.jpg", mag_s);
    imwrite("edges_simd.jpg",   mag_v);
    imwrite("edges_omp.jpg",    mag_o);
    imwrite("edges_omp_simd.jpg", mag_ov);

    return 0;
}

//g++ Edges.cpp -o Edges.exe -O3 -fopenmp -march=native `pkg-config --cflags --libs opencv4`

// ================= OUTPUT ==================

// Serial:        0.0139356 s
// SIMD:          0.00790719 s
// OpenMP:        0.0793002 s
// OMP+SIMD:      0.00510322 s
// 
// Speedup SIMD:       1.76239x
// Speedup OpenMP:     0.175732x
// Speedup OMP+SIMD:   2.73074x