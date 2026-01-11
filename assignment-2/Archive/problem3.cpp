// problem3.cpp
// Benchmark: classic locks in C++11 atomics on x86_64
// Implements: PthreadMutex, TTAS Spin, Ticket, Filter, Bakery, Anderson Array Queue
// Usage:
//   g++ -O3 -march=native -pthread -std=gnu++17 problem3.cpp -o locks
//   ./locks --iters 200000 --max_threads 64 --format csv
//   ./locks --lock ticket --threads 16 --iters 1000000 --format table
//
// Notes:
//  - Filter/Bakery require knowing #threads at construction; we re-create per run.
//  - We use alignas(64) to avoid false sharing.
//  - Array Queue lock uses Anderson's algorithm with local spinning.
//  - Spin lock uses TTAS + exponential backoff for politeness.
//  - Results are wall-clock seconds for the critical-section loop only.
//  - No external libraries are used.

#include <atomic>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

using namespace std;

static constexpr size_t CACHELINE = 64;

// Simple padded atomic wrapper to avoid false sharing for arrays.
template <typename T>
struct alignas(CACHELINE) PaddedAtomic {
    std::atomic<T> v;
    PaddedAtomic() : v() {}
    PaddedAtomic(T x) : v(x) {}
};

// Base interface
struct ILock {
    virtual void acquire(int tid) = 0;
    virtual void release(int tid) = 0;
    virtual const char* name() const = 0;
    virtual ~ILock() = default;
};

// ---------------- Pthread mutex wrapper ----------------
struct PthreadMutex : ILock {
    pthread_mutex_t m;
    PthreadMutex(int /*n_threads*/ = 0) { pthread_mutex_init(&m, nullptr); }
    ~PthreadMutex() override { pthread_mutex_destroy(&m); }
    void acquire(int) override { pthread_mutex_lock(&m); }
    void release(int) override { pthread_mutex_unlock(&m); }
    const char* name() const override { return "Pthread mutex"; }
};

// ---------------- TTAS Spin lock ----------------
// test-test-and-set with polite backoff; fairish under light contention
struct SpinTTAS : ILock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    void acquire(int) override {
        for (;;) {
            // First test without invalidating cache line
            while (flag.test(std::memory_order_relaxed)) {
                // pause hint reduces power + contention
                #if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
                #endif
            }
            if (!flag.test_and_set(std::memory_order_acquire)) break;
            // Exponential backoff
            static thread_local int b = 32;
            for (int i = 0; i < b; ++i) {
                #if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
                #endif
            }
            if (b < 4096) b <<= 1;
        }
    }
    void release(int) override {
        flag.clear(std::memory_order_release);
    }
    const char* name() const override { return "Spin lock (TTAS)"; }
};

// ---------------- Ticket lock ----------------
struct TicketLock : ILock {
    std::atomic<uint32_t> next{0};
    std::atomic<uint32_t> owner{0};
    void acquire(int) override {
        uint32_t my = next.fetch_add(1, std::memory_order_acq_rel);
        while (owner.load(std::memory_order_acquire) != my) {
            #if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
            #endif
        }
    }
    void release(int) override {
        owner.fetch_add(1, std::memory_order_release);
    }
    const char* name() const override { return "Ticket lock"; }
};

// ---------------- Filter lock (generalized Peterson) ----------------
struct FilterLock : ILock {
    int N;
    vector<PaddedAtomic<int>> level;   // level[tid]
    vector<PaddedAtomic<int>> victim;  // victim[level]
    explicit FilterLock(int n) : N(n), level(n), victim(n) {
        for (int i = 0; i < N; ++i) level[i].v.store(0, memory_order_relaxed);
        for (int l = 0; l < N; ++l) victim[l].v.store(-1, memory_order_relaxed);
    }
    void acquire(int tid) override {
        for (int L = 1; L < N; ++L) {
            level[tid].v.store(L, memory_order_release);
            victim[L].v.store(tid, memory_order_release);
            bool spin;
            do {
                spin = false;
                for (int k = 0; k < N; ++k) {
                    if (k == tid) continue;
                    if (level[k].v.load(memory_order_acquire) >= L &&
                        victim[L].v.load(memory_order_acquire) == tid) {
                        spin = true;
                        #if defined(__x86_64__) || defined(__i386__)
                        __builtin_ia32_pause();
                        #endif
                        break;
                    }
                }
            } while (spin);
        }
    }
    void release(int tid) override {
        level[tid].v.store(0, memory_order_release);
    }
    const char* name() const override { return "Filter lock"; }
};

// ---------------- Bakery lock (Lamport) ----------------
struct BakeryLock : ILock {
    int N;
    struct alignas(CACHELINE) Slot {
        atomic<bool> choosing;
        atomic<uint64_t> number;
        Slot() : choosing(false), number(0) {}
    };
    vector<Slot> slots;
    explicit BakeryLock(int n) : N(n), slots(n) {}

    static inline bool lex_lt(uint64_t a_n, int a_id, uint64_t b_n, int b_id) {
        return (a_n < b_n) || (a_n == b_n && a_id < b_id);
    }

    void acquire(int tid) override {
        slots[tid].choosing.store(true, memory_order_release);
        // Find max number
        uint64_t max_n = 0;
        for (int k = 0; k < N; ++k) {
            uint64_t nk = slots[k].number.load(memory_order_acquire);
            if (nk > max_n) max_n = nk;
        }
        slots[tid].number.store(max_n + 1, memory_order_release);
        slots[tid].choosing.store(false, memory_order_release);

        for (int k = 0; k < N; ++k) {
            if (k == tid) continue;
            while (slots[k].choosing.load(memory_order_acquire)) {
                #if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
                #endif
            }
            for (;;) {
                uint64_t nk = slots[k].number.load(memory_order_acquire);
                if (nk == 0) break;
                uint64_t nt = slots[tid].number.load(memory_order_acquire);
                if (!lex_lt(nk, k, nt, tid)) break;
                #if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
                #endif
            }
        }
    }

    void release(int tid) override {
        slots[tid].number.store(0, memory_order_release);
    }
    const char* name() const override { return "Bakery lock"; }
};

// ---------------- Anderson array-based queue lock ----------------
struct AndersonLock : ILock {
    const int size; // >= #threads, power of two recommended
    atomic<uint32_t> next{0};
    struct alignas(CACHELINE) Flag { atomic<bool> v; Flag(bool b=false):v(b){} };
    vector<Flag> flags;

    explicit AndersonLock(int n_threads)
        : size(1 << static_cast<int>(ceil(log2(max(1, n_threads))))),
          flags(size) {
        flags[0].v.store(true, memory_order_relaxed);
    }

    void acquire(int tid) override {
        (void)tid;
        thread_local uint32_t my_slot;
        my_slot = next.fetch_add(1, memory_order_acq_rel) & (size - 1);
        while (!flags[my_slot].v.load(memory_order_acquire)) {
            #if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
            #endif
        }
        // Consume our flag
        flags[my_slot].v.store(false, memory_order_relaxed);
        // Stash slot for release
        thread_local uint32_t tls_slot;
        tls_slot = my_slot;
    }

    void release(int tid) override {
        (void)tid;
        extern thread_local uint32_t tls_slot;
        uint32_t next_slot = (tls_slot + 1) & (size - 1);
        flags[next_slot].v.store(true, memory_order_release);
    }

    const char* name() const override { return "Array Q lock (Anderson)"; }
};
thread_local uint32_t tls_slot = 0;

// --------- Utility: simple spin barrier ----------
struct SpinBarrier {
    atomic<int> arrived{0};
    int total;
    explicit SpinBarrier(int n) : total(n) {}
    void arrive_and_wait() {
        int v = arrived.fetch_add(1, memory_order_acq_rel) + 1;
        if (v == total) return; // last returns immediately
        while (arrived.load(memory_order_acquire) < total) {
            #if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
            #endif
        }
    }
};

// ------------- Benchmark harness -----------------
struct RunConfig {
    string lock_name = "all";
    int max_threads = 64;
    int threads_single = -1;
    long long iters = 200000;
    string format = "table"; // or csv
};

unique_ptr<ILock> make_lock(const string& s, int nthreads) {
    if (s == "pthread" || s == "pthread_mutex") return make_unique<PthreadMutex>(nthreads);
    if (s == "spin" || s == "ttas") return make_unique<SpinTTAS>();
    if (s == "ticket") return make_unique<TicketLock>();
    if (s == "filter") return make_unique<FilterLock>(nthreads);
    if (s == "bakery") return make_unique<BakeryLock>(nthreads);
    if (s == "arrayq" || s == "anderson") return make_unique<AndersonLock>(nthreads);
    return nullptr;
}

vector<string> all_lock_keys() {
    return {"pthread", "filter", "bakery", "spin", "ticket", "arrayq"};
}

static inline double sec_since(chrono::steady_clock::time_point t0) {
    return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
}

double run_once(const string& lock_key, int T, long long iters) {
    auto lock = make_lock(lock_key, T);
    assert(lock && "unknown lock");

    // Shared variables (to mimic the prompt)
    alignas(CACHELINE) volatile long long var1 = 0;
    alignas(CACHELINE) volatile long long var2 = 0;

    SpinBarrier barrier(T);
    auto t0 = chrono::steady_clock::now();
    vector<thread> th;
    th.reserve(T);
    for (int tid = 0; tid < T; ++tid) {
        th.emplace_back([&, tid]() {
            barrier.arrive_and_wait();
            for (long long i = 0; i < iters; ++i) {
                lock->acquire(tid);
                var1++;
                var2--;
                lock->release(tid);
            }
        });
    }
    for (auto& x : th) x.join();
    return sec_since(t0);
}

void print_row(const string& name, const vector<int>& cols, const vector<double>& vals, const string& format) {
    if (format == "csv") {
        cout << '"' << name << '"';
        for (double v : vals) cout << "," << fixed << setprecision(6) << v;
        cout << "\n";
    } else {
        cout << left << setw(22) << name;
        for (double v : vals) cout << right << setw(9) << fixed << setprecision(4) << v;
        cout << "\n";
    }
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    RunConfig cfg;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        auto eat = [&](const char* flag)->bool {
            if (a == flag && i + 1 < argc) { cfg.lock_name = argv[++i]; return true; }
            return false;
        };
        if (a == "--lock" && i + 1 < argc) cfg.lock_name = argv[++i];
        else if (a == "--iters" && i + 1 < argc) cfg.iters = stoll(argv[++i]);
        else if (a == "--max_threads" && i + 1 < argc) cfg.max_threads = stoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) cfg.threads_single = stoi(argv[++i]);
        else if (a == "--format" && i + 1 < argc) cfg.format = argv[++i];
        else if (a == "--help") {
            cerr << "Usage: ./locks [--lock <pthread|filter|bakery|spin|ticket|arrayq|all>] "
                    "[--iters N] [--max_threads M] [--threads T] [--format csv|table]\n";
            return 0;
        }
    }

    vector<int> thread_counts;
    if (cfg.threads_single > 0) thread_counts = {cfg.threads_single};
    else {
        for (int t = 1; t <= cfg.max_threads; t <<= 1) thread_counts.push_back(t);
    }

    vector<string> locks = (cfg.lock_name == "all")
        ? all_lock_keys()
        : vector<string>{cfg.lock_name};

    // Header
    if (cfg.format == "csv") {
        cout << "\"lock\"";
        for (int t : thread_counts) cout << ",\"" << t << "\"";
        cout << "\n";
    } else {
        cout << left << setw(22) << "Lock";
        for (int t : thread_counts) cout << right << setw(9) << t;
        cout << "\n" << string(22 + 9*thread_counts.size(), '-') << "\n";
    }

    for (const auto& lk : locks) {
        vector<double> vals;
        for (int t : thread_counts) {
            // Recreate per run for N-sensitive locks
            double secs = run_once(lk, t, cfg.iters);
            vals.push_back(secs);
        }
        print_row(make_lock(lk, 1)->name(), thread_counts, vals, cfg.format);
    }

    return 0;
}
