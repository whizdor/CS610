#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>
#include <random>

#ifdef USE_PAPI
#include <papi.h>
#endif

using std::cerr;
using std::cout;
using std::endl;
using std::uint64_t;

// Clear Cache for Accurate Timing
static inline size_t llc_kb_from_env()
{
  if (const char *s = std::getenv("LLCKB"))
  {
    long v = std::strtol(s, nullptr, 10);
    if (v > 0)
      return (size_t)v;
  }
  return 64 * 1024; // default 64 MB
}

struct CacheFlusher
{
  std::vector<uint8_t> buf;
  size_t stride; // touch one line per step

  explicit CacheFlusher(size_t kb = llc_kb_from_env(), size_t line = 64)
      : buf(kb * 1024 + 4096), stride(line) {}

  void flush()
  {
    volatile uint8_t acc = 0;
    // Pass 1: read
    for (size_t i = 0; i < buf.size(); i += stride)
      acc ^= buf[i];
    // Pass 2: write
    for (size_t i = 0; i < buf.size(); i += stride)
      buf[i] = (uint8_t)(acc + i);
    asm volatile("" ::: "memory"); // compile barrier
  }
};

// ------------ Problem Sizes -----------
#define INP_H (1 << 7)
#define INP_W (1 << 7)
#define INP_D (1 << 7)
#define FIL_H (3)
#define FIL_W (3)
#define FIL_D (3)

// ---------- Helpers ----------
static inline size_t outH() { return INP_H - FIL_H + 1; }
static inline size_t outW() { return INP_W - FIL_W + 1; }
static inline size_t outD() { return INP_D - FIL_D + 1; }

static inline size_t inpIdx(size_t i, size_t j, size_t k)
{
  return i * (size_t)INP_W * INP_D + j * (size_t)INP_D + k;
}
static inline size_t outIdx(size_t i, size_t j, size_t k,
                            size_t OW, size_t OD)
{
  return i * OW * OD + j * OD + k;
}

uint64_t checksum(const uint64_t *a, size_t n)
{
  uint64_t s = 0;
  for (size_t i = 0; i < n; ++i)
    s ^= (a[i] + 0x9e3779b97f4a7c15ULL + (s << 6) + (s >> 2));
  return s;
}

/** Cross-correlation without padding */
void cc_3d_naive(const uint64_t *input,
                 const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t *result,
                 const uint64_t outputHeight, const uint64_t outputWidth,
                 const uint64_t outputDepth)
{
  for (uint64_t i = 0; i < outputHeight; i++)
  {
    for (uint64_t j = 0; j < outputWidth; j++)
    {
      for (uint64_t k = 0; k < outputDepth; k++)
      {
        uint64_t sum = 0;
        for (uint64_t ki = 0; ki < FIL_H; ki++)
        {
          for (uint64_t kj = 0; kj < FIL_W; kj++)
          {
            for (uint64_t kk = 0; kk < FIL_D; kk++)
            {
              sum += input[(i + ki) * INP_W * INP_D + (j + kj) * INP_D +
                           (k + kk)] *
                     kernel[ki][kj][kk];
            }
          }
        }
        result[i * outputWidth * outputDepth + j * outputDepth + k] += sum;
      }
    }
  }
}

void cc_3d_blocked(const uint64_t *__restrict input,
                   const uint64_t (*__restrict kernel)[FIL_W][FIL_D],
                   uint64_t *__restrict result,
                   size_t OH, size_t OW, size_t OD,
                   int Bi, int Bj, int Bk)
{
  for (size_t ii = 0; ii < OH; ii += Bi)
  {
    const size_t iMax = std::min(OH, ii + (size_t)Bi);
    for (size_t jj = 0; jj < OW; jj += Bj)
    {
      const size_t jMax = std::min(OW, jj + (size_t)Bj);
      for (size_t kk = 0; kk < OD; kk += Bk)
      {
        const size_t kMax = std::min(OD, kk + (size_t)Bk);

        for (size_t i = ii; i < iMax; ++i)
        {
          for (size_t j = jj; j < jMax; ++j)
          {
            size_t baseOut = outIdx(i, j, kk, OW, OD);
            for (size_t k = kk; k < kMax; ++k)
            {
              uint64_t sum = 0;
              for (size_t ki = 0; ki < FIL_H; ++ki)
              {
                for (size_t kj = 0; kj < FIL_W; ++kj)
                {
                  const uint64_t *inptr = &input[inpIdx(i + ki, j + kj, k)];
                  // small inner loop tends to vectorize well
                  for (size_t kz = 0; kz < FIL_D; ++kz)
                  {
                    sum += inptr[kz] * kernel[ki][kj][kz];
                  }
                }
              }
              result[baseOut + (k - kk)] = sum;
            }
          }
        }
      }
    }
  }
}

// ---------- PAPI wrapper ----------
#ifdef USE_PAPI
#include <papi.h>
#include <functional>

static bool papi_ok = false;
static int  EventSet = PAPI_NULL;
static int  added_events[3];
static int  n_events = 0;

struct Counters { long long tot_cyc = -1, l1_dcm = -1, l2_dcm = -1; };

void papi_init() {
  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) return;
  if (PAPI_create_eventset(&EventSet) != PAPI_OK) return;

  int candidates[3] = { PAPI_TOT_CYC, PAPI_L1_DCM, PAPI_L2_DCM };
  for (int e : candidates) {
    if (PAPI_query_event(e) == PAPI_OK) {
      if (PAPI_add_event(EventSet, e) == PAPI_OK) {
        added_events[n_events++] = e;
      }
    }
  }
  papi_ok = (n_events > 0);
}

Counters papi_measure(std::function<void()> fn) {
  Counters c{};
  if (!papi_ok) { fn(); return c; }

  long long values[3] = {0,0,0};
  if (PAPI_start(EventSet) != PAPI_OK) { fn(); return c; }
  fn();
  if (PAPI_stop(EventSet, values) != PAPI_OK) return c;

  for (int i = 0; i < n_events; ++i) {
    switch (added_events[i]) {
      case PAPI_TOT_CYC: c.tot_cyc = values[i]; break;
      case PAPI_L1_DCM:  c.l1_dcm = values[i];  break;
      case PAPI_L2_DCM:  c.l2_dcm = values[i];  break;
    }
  }
  PAPI_reset(EventSet);
  return c;
}

#else
void papi_init() {}
Counters papi_measure(std::function<void()> fn)
{
  fn();
  return Counters{};
}
#endif

// ---------- Timing (returns microseconds) ----------
template <class F>
std::pair<double, Counters> time_us(F &&fn)
{
  auto wrapped = [&]()
  { fn(); };
  auto t0 = std::chrono::high_resolution_clock::now();
  Counters c = papi_measure(wrapped);
  auto t1 = std::chrono::high_resolution_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  return {us, c};
}

// ---------- Autotuner for (Bi,Bj,Bk) ----------
struct Tile
{
  int Bi, Bj, Bk;
  double us;
};

size_t l2_kb_from_env()
{
  const char *s = std::getenv("L2KB");
  if (!s)
    return 512; // conservative default per-core L2
  long v = std::strtol(s, nullptr, 10);
  return v > 0 ? (size_t)v : 512;
}

// Working-set bytes we want to keep ~within L2 (with slack)
size_t tile_working_set_bytes(int Bi, int Bj, int Bk)
{
  const size_t haloH = Bi + (FIL_H - 1);
  const size_t haloW = Bj + (FIL_W - 1);
  const size_t haloD = Bk + (FIL_D - 1);
  const size_t inBytes = haloH * haloW * haloD * sizeof(uint64_t);
  const size_t outBytes = (size_t)Bi * Bj * Bk * sizeof(uint64_t);
  const size_t kerBytes = (size_t)FIL_H * FIL_W * FIL_D * sizeof(uint64_t);
  return inBytes + outBytes + kerBytes;
}

Tile autotune(const uint64_t *input,
              const uint64_t (*kernel)[FIL_W][FIL_D],
              uint64_t *outBuf, size_t OH, size_t OW, size_t OD)
{
  const size_t L2 = l2_kb_from_env() * 1024;
  const double budget = 0.85; // leave some headroom
  std::vector<int> cands = {2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32};

  Tile best{8, 8, 8, std::numeric_limits<double>::infinity()};

  for (int Bi : cands)
    for (int Bj : cands)
      for (int Bk : cands)
      {
        if (Bi <= 0 || Bj <= 0 || Bk <= 0)
          continue;
        if ((size_t)Bi > OH || (size_t)Bj > OW || (size_t)Bk > OD)
          continue;

        size_t ws = tile_working_set_bytes(Bi, Bj, Bk);
        if (ws > (size_t)(budget * L2))
          continue;

        // Warm up and measure one pass
        std::fill(outBuf, outBuf + OH * OW * OD, 0);
        auto [us1, _c1] = time_us([&]()
                                  { cc_3d_blocked(input, kernel, outBuf, OH, OW, OD, Bi, Bj, Bk); });

        if (us1 < best.us)
          best = Tile{Bi, Bj, Bk, us1};
      }
  return best;
}

int main()
{
  CacheFlusher broom;
  // -------- Allocate & initialize --------
  const size_t H = INP_H, W = INP_W, D = INP_D;
  const size_t OH = outH(), OW = outW(), OD = outD();
  const size_t IN_SZ = H * W * D, OUT_SZ = OH * OW * OD;

  auto *input = new uint64_t[IN_SZ];
  std::fill_n(input, IN_SZ, 1ULL);

  uint64_t kernel[FIL_H][FIL_W][FIL_D] = {
      {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
      {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
      {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}}};

  auto *out_naive = new uint64_t[OUT_SZ];
  auto *out_block = new uint64_t[OUT_SZ];

  papi_init();

  // -------- Run naïve (5 runs, average) --------
  double naive_sum_us = 0.0;
  long long naive_l1 = 0, naive_l2 = 0;
  int papi_runs_n = 0;

  for (int r = 0; r < 5; ++r)
  {
    std::fill(out_naive, out_naive + OUT_SZ, 0);
    broom.flush();
    auto [us, c] = time_us([&]()
                           { cc_3d_naive(input, kernel, out_naive, OH, OW, OD); });
    naive_sum_us += us;
#ifdef USE_PAPI
    if (c.l1_dcm >= 0)
    {
      naive_l1 += c.l1_dcm;
      naive_l2 += c.l2_dcm;
      ++papi_runs_n;
    }
#endif
  }
  const double naive_avg_us = naive_sum_us / 5.0;
  const uint64_t naive_chk = checksum(out_naive, OUT_SZ);

  // -------- Autotune blocked tile --------
  Tile best = autotune(input, kernel, out_block, OH, OW, OD);

  // -------- Run blocked with best tile (5 runs, average) --------
  double blocked_sum_us = 0.0;
  long long blocked_l1 = 0, blocked_l2 = 0;
  int papi_runs_b = 0;

  for (int r = 0; r < 5; ++r)
  {
    std::fill(out_block, out_block + OUT_SZ, 0);
    broom.flush();
    auto [us, c] = time_us([&]()
                           { cc_3d_blocked(input, kernel, out_block, OH, OW, OD,
                                           best.Bi, best.Bj, best.Bk); });
    blocked_sum_us += us;
#ifdef USE_PAPI
    if (c.l1_dcm >= 0)
    {
      blocked_l1 += c.l1_dcm;
      blocked_l2 += c.l2_dcm;
      ++papi_runs_b;
    }
#endif
  }
  const double blocked_avg_us = blocked_sum_us / 5.0;
  const uint64_t blocked_chk = checksum(out_block, OUT_SZ);

  // -------- Correctness check --------
  if (naive_chk != blocked_chk)
  {
    cerr << "[ERROR] Results mismatch! checksums: naive=" << naive_chk
         << " blocked=" << blocked_chk << endl;
    return 1;
  }

  // -------- Report --------
  const double ops_per_out = (double)FIL_H * FIL_W * FIL_D * 2.0; // mul+add per tap ≈ 2 ops
  const double gops = (OUT_SZ * ops_per_out) / 1e9;
  cout << std::fixed << std::setprecision(2);
  cout << "Output size: " << OH << " x " << OW << " x " << OD
       << "  (elements: " << OUT_SZ << ")\n";
  cout << "Kernel size: " << FIL_H << " x " << FIL_W << " x " << FIL_D << "\n";
  cout << "L2 budget (KB): " << l2_kb_from_env() << "  (autotune slack 85%)\n\n";

  cout << "[Naïve]   avg time: " << naive_avg_us / 1000.0 << " ms,  approx throughput: "
       << (gops / (naive_avg_us / 1e6)) << " GOPS\n";
#ifdef USE_PAPI
  if (papi_runs_n)
  {
    cout << "          L1 DCM: " << (naive_l1 / papi_runs_n)
         << "   L2 DCM: " << (naive_l2 / papi_runs_n) << "\n";
  }
#endif

  cout << "[Blocked] avg time: " << blocked_avg_us / 1000.0 << " ms,  approx throughput: "
       << (gops / (blocked_avg_us / 1e6)) << " GOPS\n";
  cout << "          tile (Bi,Bj,Bk) = (" << best.Bi << "," << best.Bj << "," << best.Bk
       << "),  one-shot autotune time for best: " << best.us / 1000.0 << " ms\n";
#ifdef USE_PAPI
  if (papi_runs_b)
  {
    cout << "          L1 DCM: " << (blocked_l1 / papi_runs_b)
         << "   L2 DCM: " << (blocked_l2 / papi_runs_b) << "\n";
  }
#endif

  cout << "Checksum: " << naive_chk << " (naïve == blocked)\n";

  delete[] input;
  delete[] out_naive;
  delete[] out_block;
  return EXIT_SUCCESS;
}
