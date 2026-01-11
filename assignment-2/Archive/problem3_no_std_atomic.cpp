// problem3_no_std_atomic.cpp
// Lock benchmarks WITHOUT std::atomic — uses GCC/Clang __atomic/__sync intrinsics.
// Implements: Pthread mutex, TTAS Spin, Ticket, Filter, Bakery, Anderson Array Q
// Build: g++ -O3 -march=native -pthread -std=gnu++17 problem3_no_std_atomic.cpp -o locks_nosa
// Run:   ./locks_nosa
//
// Notes:
//  - NUM_THREADS is compile-time here to keep arrays static for Filter/Bakery.
//  - Replace N or NUM_THREADS as needed.
//  - We keep your pthread_barrier-based harness and sum per-thread runtimes like your original.
//  - Units fix: we report milliseconds (ms).

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <pthread.h>

using std::cerr;
using std::cout;
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

#define N (1000000)      // iterations per thread (tune for runtime)
#define NUM_THREADS (8)  // threads (powers of 2 in your table)

// ------------------------------------------------------------------
// Helpers: atomics via GCC builtins + CPU relax
// ------------------------------------------------------------------
static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    asm volatile("nop");
#endif
}

// Load/store/fetch-add wrappers for clarity
template <typename T>
static inline T a_load(const volatile T* p, int mo) {
    return __atomic_load_n(p, mo);
}
template <typename T>
static inline void a_store(volatile T* p, T v, int mo) {
    __atomic_store_n(p, v, mo);
}
template <typename T>
static inline T a_faa(volatile T* p, T inc, int mo) {
    return __atomic_fetch_add(p, inc, mo);
}
template <typename T>
static inline bool a_cas(volatile T* p, T& expected, T desired, int mo_s, int mo_f) {
    return __atomic_compare_exchange_n(p, &expected, desired, false, mo_s, mo_f);
}

// ------------------------------------------------------------------
// Shared variables protected by locks
// ------------------------------------------------------------------
volatile uint64_t var1 = 0;
volatile uint64_t var2 = (uint64_t)N * (uint64_t)NUM_THREADS + 1;

// Abstract base class
class LockBase {
public:
    virtual void acquire(uint16_t tid) = 0;
    virtual void release(uint16_t tid) = 0;
    virtual ~LockBase() = default;
};

typedef struct thr_args {
    uint16_t m_id;
    LockBase* m_lock;
} ThreadArgs;

// ------------------------------------------------------------------
// Pthread mutex baseline
// ------------------------------------------------------------------
class PthreadMutex : public LockBase {
public:
    void acquire(uint16_t) override { pthread_mutex_lock(&lock); }
    void release(uint16_t) override { pthread_mutex_unlock(&lock); }
private:
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
};

// ------------------------------------------------------------------
// Spin (TTAS) using __atomic_test_and_set / __atomic_clear
// ------------------------------------------------------------------
class SpinLock : public LockBase {
private:
    // __atomic_test_and_set operates on a byte
    volatile bool flag = false;
public:
    void acquire(uint16_t) override {
        for (;;) {
            while (a_load(&flag, __ATOMIC_RELAXED)) cpu_relax();
            if (!__atomic_test_and_set(&flag, __ATOMIC_ACQUIRE)) break;
            for (int k = 0; k < 64; ++k) cpu_relax();
        }
    }
    void release(uint16_t) override {
        __atomic_clear(&flag, __ATOMIC_RELEASE);
    }
};

// ------------------------------------------------------------------
// Ticket lock (FIFO)
// ------------------------------------------------------------------
class TicketLock : public LockBase {
private:
    volatile uint32_t next  = 0;
    volatile uint32_t owner = 0;
public:
    void acquire(uint16_t) override {
        uint32_t my = a_faa(&next, (uint32_t)1, __ATOMIC_ACQ_REL);
        while (a_load(&owner, __ATOMIC_ACQUIRE) != my) cpu_relax();
    }
    void release(uint16_t) override {
        a_faa(&owner, (uint32_t)1, __ATOMIC_RELEASE);
    }
};

// ------------------------------------------------------------------
// Filter lock (generalized Peterson). NUM_THREADS fixed at compile time.
// ------------------------------------------------------------------
class FilterLock : public LockBase {
private:
    alignas(64) volatile int level[NUM_THREADS];
    alignas(64) volatile int victim[NUM_THREADS];
public:
    FilterLock() {
        for (int i = 0; i < NUM_THREADS; ++i) level[i] = 0;
        for (int i = 0; i < NUM_THREADS; ++i) victim[i] = -1;
    }
    void acquire(uint16_t tid) override {
        for (int L = 1; L < NUM_THREADS; ++L) {
            a_store(&level[tid], L, __ATOMIC_RELEASE);
            a_store(&victim[L], (int)tid, __ATOMIC_RELEASE);
            bool spin;
            do {
                spin = false;
                for (int k = 0; k < NUM_THREADS; ++k) {
                    if (k == tid) continue;
                    int lk = a_load(&level[k], __ATOMIC_ACQUIRE);
                    int vL = a_load(&victim[L], __ATOMIC_ACQUIRE);
                    if (lk >= L && vL == (int)tid) { spin = true; cpu_relax(); break; }
                }
            } while (spin);
        }
    }
    void release(uint16_t tid) override {
        a_store(&level[tid], 0, __ATOMIC_RELEASE);
    }
};

// ------------------------------------------------------------------
// Bakery lock (Lamport). NUM_THREADS fixed.
// ------------------------------------------------------------------
class BakeryLock : public LockBase {
private:
    alignas(64) volatile bool     choosing[NUM_THREADS];
    alignas(64) volatile uint64_t number[NUM_THREADS];
    static inline bool before(uint64_t a_n, int a_id, uint64_t b_n, int b_id) {
        return (a_n < b_n) || (a_n == b_n && a_id < b_id);
    }
public:
    BakeryLock() {
        for (int i = 0; i < NUM_THREADS; ++i) { choosing[i] = false; number[i] = 0; }
    }
    void acquire(uint16_t tid) override {
        a_store(&choosing[tid], true, __ATOMIC_RELEASE);
        uint64_t maxn = 0;
        for (int k = 0; k < NUM_THREADS; ++k) {
            uint64_t nk = a_load(&number[k], __ATOMIC_ACQUIRE);
            if (nk > maxn) maxn = nk;
        }
        a_store(&number[tid], maxn + 1, __ATOMIC_RELEASE);
        a_store(&choosing[tid], false, __ATOMIC_RELEASE);

        for (int k = 0; k < NUM_THREADS; ++k) {
            if (k == tid) continue;
            while (a_load(&choosing[k], __ATOMIC_ACQUIRE)) cpu_relax();
            for (;;) {
                uint64_t nk = a_load(&number[k], __ATOMIC_ACQUIRE);
                if (nk == 0) break;
                uint64_t nt = a_load(&number[tid], __ATOMIC_ACQUIRE);
                if (!before(nk, k, nt, tid)) break;
                cpu_relax();
            }
        }
    }
    void release(uint16_t tid) override {
        a_store(&number[tid], (uint64_t)0, __ATOMIC_RELEASE);
    }
};

// ------------------------------------------------------------------
// Anderson array-based queue lock
// ------------------------------------------------------------------
class ArrayQLock : public LockBase {
private:
    static constexpr uint32_t Sz = 1u << 3; // 8 slots (>= NUM_THREADS), power of two
    alignas(64) volatile bool flags[Sz];
    volatile uint32_t next = 0;
    // Per-thread slot index
    alignas(64) volatile uint32_t slot_of[NUM_THREADS];
public:
    ArrayQLock() {
        for (uint32_t i = 0; i < Sz; ++i) flags[i] = false;
        flags[0] = true;
        for (int i = 0; i < NUM_THREADS; ++i) slot_of[i] = 0;
    }
    void acquire(uint16_t tid) override {
        uint32_t my = a_faa(&next, (uint32_t)1, __ATOMIC_ACQ_REL) & (Sz - 1);
        slot_of[tid] = my;
        while (!a_load(&flags[my], __ATOMIC_ACQUIRE)) cpu_relax();
        a_store(&flags[my], false, __ATOMIC_RELAXED);
    }
    void release(uint16_t tid) override {
        uint32_t nxt = (slot_of[tid] + 1) & (Sz - 1);
        a_store(&flags[nxt], true, __ATOMIC_RELEASE);
    }
};

// ------------------------------------------------------------------
// Benchmark plumbing (no std::atomic).
// We sum per-thread durations using an atomic fetch_add on a plain uint64_t.
// ------------------------------------------------------------------
volatile uint64_t sync_time_ms = 0;

inline void critical_section() {
    // Intentionally not atomic; protected by the lock
    var1++;
    var2--;
}

pthread_barrier_t g_barrier;

void* thrBody(void* arguments) {
    ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);

    pthread_barrier_wait(&g_barrier);

    HRTimer start = HR::now();
    for (int i = 0; i < N; i++) {
        tmp->m_lock->acquire(tmp->m_id);
        critical_section();
        tmp->m_lock->release(tmp->m_id);
    }
    HRTimer end = HR::now();
    uint64_t duration = (uint64_t)duration_cast<milliseconds>(end - start).count();

    __atomic_fetch_add(&sync_time_ms, duration, __ATOMIC_RELAXED);
    return nullptr;
}

static void run_one(const char* name, LockBase* lock_obj) {
    var1 = 0;
    var2 = (uint64_t)N * (uint64_t)NUM_THREADS + 1;
    __atomic_store_n(&sync_time_ms, (uint64_t)0, __ATOMIC_RELAXED);

    pthread_t tid[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];
    for (uint16_t i = 0; i < NUM_THREADS; ++i) {
        args[i].m_id = i;
        args[i].m_lock = lock_obj;
        int err = pthread_create(&tid[i], nullptr, thrBody, (void*)(args + i));
        if (err != 0) { cerr << "Thread cannot be created: " << strerror(err) << "\n"; exit(1); }
    }
    for (int i = 0; i < NUM_THREADS; ++i) pthread_join(tid[i], nullptr);

    assert(var1 == (uint64_t)N * (uint64_t)NUM_THREADS && "var1 mismatch");
    assert(var2 == 1 && "var2 mismatch");

    cout << name << ": summed thread time (ms): " << sync_time_ms << "\n";
}

int main() {
    int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
    if (error != 0) { cerr << "Error in barrier init.\n"; return 1; }

    cout << "N=" << N << ", threads=" << NUM_THREADS << "\n";

    PthreadMutex pthread_lock;
    run_one("Pthread mutex", &pthread_lock);

    FilterLock filter_lock;
    run_one("Filter lock", &filter_lock);

    BakeryLock bakery_lock;
    run_one("Bakery lock", &bakery_lock);

    SpinLock spin_lock;
    run_one("Spin lock (TTAS)", &spin_lock);

    TicketLock ticket_lock;
    run_one("Ticket lock", &ticket_lock);

    ArrayQLock arrayq_lock;
    run_one("Array Q lock", &arrayq_lock);

    pthread_barrier_destroy(&g_barrier);
    return 0;
}
