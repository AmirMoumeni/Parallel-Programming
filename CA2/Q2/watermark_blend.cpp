#include <iostream>
#include <chrono>
#include <immintrin.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool read_image(const char* filename, float*& r, float*& g, float*& b, int& w, int& h) {
    int c;
    unsigned char* data = stbi_load(filename, &w, &h, &c, 3);
    if (!data) return false;

    int size = w * h;
    r = new float[size];
    g = new float[size];
    b = new float[size];

    for (int i = 0; i < size; ++i) {
        r[i] = data[i * 3 + 0];
        g[i] = data[i * 3 + 1];
        b[i] = data[i * 3 + 2];
    }
    stbi_image_free(data);
    return true;
}

void rebuild_image(const char* filename, float* r, float* g, float* b, int w, int h) {
    unsigned char* out = new unsigned char[w * h * 3];
    for (int i = 0; i < w * h; ++i) {
        out[i * 3 + 0] = (unsigned char)std::min(255.0f, std::max(0.0f, r[i]));
        out[i * 3 + 1] = (unsigned char)std::min(255.0f, std::max(0.0f, g[i]));
        out[i * 3 + 2] = (unsigned char)std::min(255.0f, std::max(0.0f, b[i]));
    }
    stbi_write_jpg(filename, w, h, 3, out, 100);
    delete[] out;
}

void resize_image(float* src_r, float* src_g, float* src_b,
                  int src_w, int src_h,
                  float*& dst_r, float*& dst_g, float*& dst_b,
                  int dst_w, int dst_h)
{
    dst_r = new float[dst_w * dst_h];
    dst_g = new float[dst_w * dst_h];
    dst_b = new float[dst_w * dst_h];

    for (int y = 0; y < dst_h; ++y) {
        int src_y = y * src_h / dst_h;
        for (int x = 0; x < dst_w; ++x) {
            int src_x = x * src_w / dst_w;
            int dst_idx = y * dst_w + x;
            int src_idx = src_y * src_w + src_x;
            dst_r[dst_idx] = src_r[src_idx];
            dst_g[dst_idx] = src_g[src_idx];
            dst_b[dst_idx] = src_b[src_idx];
        }
    }
}

void blend_serial(float* rb, float* gb, float* bb,
                  float* rw, float* gw, float* bw,
                  float* ro, float* go, float* bo,
                  int w, int h)
{
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float alpha = (float)(x + y) / (float)(w + h);
            int idx = y * w + x;
            ro[idx] = alpha * rw[idx] + (1 - alpha) * rb[idx];
            go[idx] = alpha * gw[idx] + (1 - alpha) * gb[idx];
            bo[idx] = alpha * bw[idx] + (1 - alpha) * bb[idx];
        }
    }
}

void blend_sse(float* rb, float* gb, float* bb,
               float* rw, float* gw, float* bw,
               float* ro, float* go, float* bo,
               int w, int h)
{
    __m128 one = _mm_set1_ps(1.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; x += 4) {
            float alpha_f[4] = {0};
            for (int i = 0; i < 4 && (x + i) < w; ++i)
                alpha_f[i] = (float)(x + i + y) / (float)(w + h);

            __m128 alpha = _mm_loadu_ps(alpha_f);
            __m128 inv_alpha = _mm_sub_ps(one, alpha);

            int idx = y * w + x;
            __m128 rbv = _mm_loadu_ps(&rb[idx]);
            __m128 gbv = _mm_loadu_ps(&gb[idx]);
            __m128 bbv = _mm_loadu_ps(&bb[idx]);

            __m128 rwv = _mm_loadu_ps(&rw[idx]);
            __m128 gwv = _mm_loadu_ps(&gw[idx]);
            __m128 bwv = _mm_loadu_ps(&bw[idx]);

            __m128 rout = _mm_add_ps(_mm_mul_ps(alpha, rwv), _mm_mul_ps(inv_alpha, rbv));
            __m128 gout = _mm_add_ps(_mm_mul_ps(alpha, gwv), _mm_mul_ps(inv_alpha, gbv));
            __m128 bout = _mm_add_ps(_mm_mul_ps(alpha, bwv), _mm_mul_ps(inv_alpha, bbv));

            _mm_storeu_ps(&ro[idx], rout);
            _mm_storeu_ps(&go[idx], gout);
            _mm_storeu_ps(&bo[idx], bout);
        }
    }
}

int main() {
    float *rb, *gb, *bb;         
    float *rw, *gw, *bw;      
    int w_base, h_base, w_wm, h_wm;

    if (!read_image("base.jpg", rb, gb, bb, w_base, h_base) ||
        !read_image("watermark.png", rw, gw, bw, w_wm, h_wm)) {
        std::cerr << "Error loading images!\n";
        return -1;
    }

    float *rw_r, *gw_r, *bw_r;
    if (w_base != w_wm || h_base != h_wm) {
        resize_image(rw, gw, bw, w_wm, h_wm, rw_r, gw_r, bw_r, w_base, h_base);
    } else {
        rw_r = rw; gw_r = gw; bw_r = bw;
    }

    int size = w_base * h_base;
    float *ro_s = new float[size], *go_s = new float[size], *bo_s = new float[size];
    float *ro_v = new float[size], *go_v = new float[size], *bo_v = new float[size];

    auto t1 = std::chrono::high_resolution_clock::now();
    blend_serial(rb, gb, bb, rw_r, gw_r, bw_r, ro_s, go_s, bo_s, w_base, h_base);
    auto t2 = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration<double>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    blend_sse(rb, gb, bb, rw_r, gw_r, bw_r, ro_v, go_v, bo_v, w_base, h_base);
    t2 = std::chrono::high_resolution_clock::now();
    double sse_time = std::chrono::duration<double>(t2 - t1).count();

    std::cout << "Serial time: " << serial_time << " s\n";
    std::cout << "SSE time:    " << sse_time << " s\n";
    std::cout << "Speedup:     " << serial_time / sse_time << "x\n";

    rebuild_image("output_serial.jpg", ro_s, go_s, bo_s, w_base, h_base);
    rebuild_image("output_sse.jpg", ro_v, go_v, bo_v, w_base, h_base);

    // Free memory
    delete[] rb; delete[] gb; delete[] bb;
    delete[] rw; delete[] gw; delete[] bw;
    if (rw_r != rw) { delete[] rw_r; delete[] gw_r; delete[] bw_r; }
    delete[] ro_s; delete[] go_s; delete[] bo_s;
    delete[] ro_v; delete[] go_v; delete[] bo_v;

    return 0;
}
