#include <iostream>
#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <chrono>
using namespace std;
using namespace cv;

Mat convolve_serial(const Mat& src, const float kernel[3][3]) {
    Mat dst = Mat::zeros(src.size(), CV_32F);
    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sum += src.at<float>(y + ky, x + kx) * kernel[ky + 1][kx + 1];
                }
            }
            dst.at<float>(y, x) = sum;
        }
    }
    return dst;
}

Mat convolve_simd(const Mat& src, const float kernel[3][3]) {
    Mat dst = Mat::zeros(src.size(), CV_32F);
    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; x += 8) { 
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

int main() {
    Mat img = imread("base.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Image not found!" << endl;
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
        {0, 0, 0},
        {1, 2, 1}
    };

    // ---------------- Serial ----------------
    auto t1 = chrono::high_resolution_clock::now();
    Mat blurred_s = convolve_serial(img, gaussian_blur);
    Mat edge_x_s = convolve_serial(blurred_s, sobel_x);
    Mat edge_y_s = convolve_serial(blurred_s, sobel_y);
    Mat magnitude_s;
    magnitude_s = edge_x_s.mul(edge_x_s) + edge_y_s.mul(edge_y_s);
    sqrt(magnitude_s, magnitude_s);
    magnitude_s.convertTo(magnitude_s, CV_8U);
    auto t2 = chrono::high_resolution_clock::now();
    double time_serial = chrono::duration<double>(t2 - t1).count();

    // ---------------- SIMD ----------------
    auto t3 = chrono::high_resolution_clock::now();
    Mat blurred_p = convolve_simd(img, gaussian_blur);
    Mat edge_x_p = convolve_simd(blurred_p, sobel_x);
    Mat edge_y_p = convolve_simd(blurred_p, sobel_y);
    Mat magnitude_p;
    magnitude_p = edge_x_p.mul(edge_x_p) + edge_y_p.mul(edge_y_p);
    sqrt(magnitude_p, magnitude_p);
    magnitude_p.convertTo(magnitude_p, CV_8U);
    auto t4 = chrono::high_resolution_clock::now();
    double time_simd = chrono::duration<double>(t4 - t3).count();

    double speedup = time_serial / time_simd;

    cout << "Serial time: " << time_serial << " s\n";
    cout << "SIMD time:   " << time_simd << " s\n";
    cout << "Speedup:     " << speedup << "x\n";

    imwrite("edges_serial.jpg", magnitude_s);
    imwrite("edges_simd.jpg", magnitude_p);

    return 0;
}
