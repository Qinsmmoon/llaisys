#include "add_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstdint>

#if defined(_MSC_VER)
    // MSVC on Windows
    #include <intrin.h>
    #include <immintrin.h>
#else
    // GCC/Clang
    #include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaisys::ops::cpu {

// 并行阈值：当元素数量超过16384时启用OpenMP并行
static constexpr size_t PARALLEL_THRESHOLD = 1 << 14; // 16384 elements

// Generic F32 implementation with optional OpenMP parallelization
template <typename T>
inline void add_f32(T *c, const T *a, const T *b, size_t numel) {
#if defined(_OPENMP)
    if (numel >= PARALLEL_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)numel; ++i) {
            c[i] = a[i] + b[i];
        }
    } else
#endif
    {
        for (size_t i = 0; i < numel; i++) {
            c[i] = a[i] + b[i];
        }
    }
}

// Fallback generic FP16: elementwise cast via utils (scalar), with OpenMP
void add_generic_fp16(llaisys::fp16_t *c, const llaisys::fp16_t *a,
                      const llaisys::fp16_t *b, size_t numel) {
#if defined(_OPENMP)
    if (numel >= PARALLEL_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)numel; ++i) {
            c[i] = llaisys::utils::cast<llaisys::fp16_t>(
                llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i])
            );
        }
    } else
#endif
    {
        for (size_t i = 0; i < numel; ++i) {
            c[i] = llaisys::utils::cast<llaisys::fp16_t>(
                llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i])
            );
        }
    }
}

// Fallback generic BF16: elementwise cast via utils (scalar), with OpenMP
void add_generic_bf16(llaisys::bf16_t *c, const llaisys::bf16_t *a,
                      const llaisys::bf16_t *b, size_t numel) {
#if defined(_OPENMP)
    if (numel >= PARALLEL_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)numel; ++i) {
            c[i] = llaisys::utils::cast<llaisys::bf16_t>(
                llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i])
            );
        }
    } else
#endif
    {
        for (size_t i = 0; i < numel; ++i) {
            c[i] = llaisys::utils::cast<llaisys::bf16_t>(
                llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i])
            );
        }
    }
}

#if defined(__AVX2__) || defined(_MSC_VER)

// Helper: tail scalar processors
static inline void add_tail_fp16(llaisys::fp16_t *c, const llaisys::fp16_t *a,
                                 const llaisys::fp16_t *b, size_t start, size_t numel) {
    for (size_t i = start; i < numel; ++i) {
        c[i] = llaisys::utils::cast<llaisys::fp16_t>(
            llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i])
        );
    }
}

static inline void add_tail_bf16(llaisys::bf16_t *c, const llaisys::bf16_t *a,
                                 const llaisys::bf16_t *b, size_t start, size_t numel) {
    for (size_t i = start; i < numel; ++i) {
        c[i] = llaisys::utils::cast<llaisys::bf16_t>(
            llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i])
        );
    }
}

//
// AVX2 + optional F16C FP16 implementation
//
#if defined(__F16C__) || defined(ENABLE_F16C)
void add_avx2_fp16(llaisys::fp16_t *c, const llaisys::fp16_t *a,
                   const llaisys::fp16_t *b, size_t numel) {
    const size_t stride = 16;
#if defined(_OPENMP)
    if (numel >= PARALLEL_THRESHOLD) {
        #pragma omp parallel
        {
            size_t nthr = omp_get_num_threads();
            size_t tid = omp_get_thread_num();
            size_t chunk = (numel + nthr - 1) / nthr;
            size_t s = tid * chunk;
            size_t e = s + chunk;
            if (e > numel) e = numel;
            size_t j = s;

            for (; j + stride - 1 < e; j += stride) {
                const __m128i v_a_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j));
                const __m128i v_a_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j + 8));
                const __m128i v_b_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j));
                const __m128i v_b_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j + 8));

                const __m256 fa0 = _mm256_cvtph_ps(v_a_lo);
                const __m256 fa1 = _mm256_cvtph_ps(v_a_hi);
                const __m256 fb0 = _mm256_cvtph_ps(v_b_lo);
                const __m256 fb1 = _mm256_cvtph_ps(v_b_hi);

                const __m256 fc0 = _mm256_add_ps(fa0, fb0);
                const __m256 fc1 = _mm256_add_ps(fa1, fb1);

                const __m128i rc_lo = _mm256_cvtps_ph(fc0, 0);
                const __m128i rc_hi = _mm256_cvtps_ph(fc1, 0);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c + j), rc_lo);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c + j + 8), rc_hi);
            }

            add_tail_fp16(c, a, b, j, e);
        }
        return;
    }
#endif

    size_t i = 0;
    // 循环条件：确保还有足够的数据（stride=16个元素）
    for (; i + stride - 1 < numel; i += stride) {
        // 从数组a的位置i加载16字节（8个float16）到寄存器
        const __m128i v_a_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        // 加载接下来的8个float16（i+8位置）
        const __m128i v_a_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 8));

        const __m128i v_b_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        const __m128i v_b_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 8));

        // 将8个float16转换为8个float32（存入256位寄存器）
        const __m256 fa0 = _mm256_cvtph_ps(v_a_lo);
        const __m256 fa1 = _mm256_cvtph_ps(v_a_hi);
        const __m256 fb0 = _mm256_cvtph_ps(v_b_lo);
        const __m256 fb1 = _mm256_cvtph_ps(v_b_hi);

        // 两组数据分别相加（float32运算）
        const __m256 fc0 = _mm256_add_ps(fa0, fb0);
        const __m256 fc1 = _mm256_add_ps(fa1, fb1);

        // 将float32转换回float16（第二个参数0表示默认舍入模式）
        const __m128i rc_lo = _mm256_cvtps_ph(fc0, 0);
        const __m128i rc_hi = _mm256_cvtps_ph(fc1, 0);

        // 将16个float16结果存回c数组
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i), rc_lo);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i + 8), rc_hi);
    }
    add_tail_fp16(c, a, b, i, numel);
}
#else
// If F16C not available, fallback to generic
void add_avx2_fp16(llaisys::fp16_t *c, const llaisys::fp16_t *a,
                   const llaisys::fp16_t *b, size_t numel) {
    add_generic_fp16(c, a, b, numel);
}
#endif // F16C


//
// AVX2 BF16 implementation with round-to-nearest-even
//
void add_avx2_bf16(llaisys::bf16_t *c, const llaisys::bf16_t *a,
                   const llaisys::bf16_t *b, size_t numel) {
    const size_t stride = 16;

#if defined(_OPENMP)
    if (numel >= PARALLEL_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t base = 0; base < (ptrdiff_t)numel; base += stride) {
            size_t i = base;
            size_t lim = (size_t)base + stride;
            if (lim > numel) lim = numel;
            size_t n = lim - i;

            size_t j = 0;
            for (; j + 7 < n; j += 8) {
                __m128i va16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + j));
                __m128i vb16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + j));

                __m256i va32 = _mm256_cvtepu16_epi32(va16);
                __m256i vb32 = _mm256_cvtepu16_epi32(vb16);

                va32 = _mm256_slli_epi32(va32, 16);
                vb32 = _mm256_slli_epi32(vb32, 16);

                __m256 fa = _mm256_castsi256_ps(va32);
                __m256 fb = _mm256_castsi256_ps(vb32);

                __m256 fc = _mm256_add_ps(fa, fb);

                // reinterpret to int bits
                __m256i fi = _mm256_castps_si256(fc);

                // rounding-to-nearest-even:
                // bias = 0x7FFF, plus lowest retained bit ((fi >> 16) & 1)
                __m256i bias = _mm256_set1_epi32(0x7FFF);
                __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(fi, 16), _mm256_set1_epi32(1));
                __m256i tmp = _mm256_add_epi32(_mm256_add_epi32(fi, bias), lsb);
                __m256i fri = _mm256_srli_epi32(tmp, 16); // final 16-bit bf16 bits in low 16 of each 32

                __m128i lo = _mm256_castsi256_si128(fri);
                __m128i hi = _mm256_extracti128_si256(fri, 1);
                __m128i packed = _mm_packus_epi32(lo, hi);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i + j), packed);
            }

            for (size_t t = j; t < n; ++t) {
                size_t idx = i + t;
                c[idx] = llaisys::utils::cast<llaisys::bf16_t>(
                    llaisys::utils::cast<float>(a[idx]) + llaisys::utils::cast<float>(b[idx])
                );
            }
        }
        return;
    }
#endif

    // single-threaded path
    size_t i = 0;
    for (; i + stride - 1 < numel; i += stride) {
        for (size_t j = 0; j < stride; j += 8) {
            __m128i va16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + j));
            __m128i vb16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + j));

            __m256i va32 = _mm256_cvtepu16_epi32(va16);
            __m256i vb32 = _mm256_cvtepu16_epi32(vb16);

            va32 = _mm256_slli_epi32(va32, 16);
            vb32 = _mm256_slli_epi32(vb32, 16);

            __m256 fa = _mm256_castsi256_ps(va32);
            __m256 fb = _mm256_castsi256_ps(vb32);

            __m256 fc = _mm256_add_ps(fa, fb);
            __m256i fi = _mm256_castps_si256(fc);

            __m256i bias = _mm256_set1_epi32(0x7FFF);
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(fi, 16), _mm256_set1_epi32(1));
            __m256i tmp = _mm256_add_epi32(_mm256_add_epi32(fi, bias), lsb);
            __m256i fri = _mm256_srli_epi32(tmp, 16);

            __m128i lo = _mm256_castsi256_si128(fri);
            __m128i hi = _mm256_extracti128_si256(fri, 1);
            __m128i packed = _mm_packus_epi32(lo, hi);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i + j), packed);
        }
    }

    // tail
    add_tail_bf16(c, a, b, i, numel);
}

#endif // __AVX2__ || _MSC_VER

void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_f32(reinterpret_cast<float *>(c),
                reinterpret_cast<const float *>(a),
                reinterpret_cast<const float *>(b), numel);
        break;

    case LLAISYS_DTYPE_F16:
#if defined(__AVX2__) || defined(_MSC_VER)
        add_avx2_fp16(reinterpret_cast<llaisys::fp16_t *>(c),
                      reinterpret_cast<const llaisys::fp16_t *>(a),
                      reinterpret_cast<const llaisys::fp16_t *>(b), numel);
#else
        add_generic_fp16(reinterpret_cast<llaisys::fp16_t *>(c),
                         reinterpret_cast<const llaisys::fp16_t *>(a),
                         reinterpret_cast<const llaisys::fp16_t *>(b), numel);
#endif
        break;

    case LLAISYS_DTYPE_BF16:
#if defined(__AVX2__) || defined(_MSC_VER)
        add_avx2_bf16(reinterpret_cast<llaisys::bf16_t *>(c),
                      reinterpret_cast<const llaisys::bf16_t *>(a),
                      reinterpret_cast<const llaisys::bf16_t *>(b), numel);
#else
        add_generic_bf16(reinterpret_cast<llaisys::bf16_t *>(c),
                         reinterpret_cast<const llaisys::bf16_t *>(a),
                         reinterpret_cast<const llaisys::bf16_t *>(b), numel);
#endif
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu