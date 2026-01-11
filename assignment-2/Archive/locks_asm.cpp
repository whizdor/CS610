// locks_asm.cpp
// Lock implementations using ONLY inline asm primitives (no <atomic>, no __atomic builtins).
// Provided primitives: cas_primitive, fai_primitive, add_fetch_primitive.
// Implements: Pthread mutex, Spin (TTAS), Ticket, Filter, Bakery, Anderson Array-Q.
// Keeps your pthread barrier harness and per-lock runs.
//
// Build:
//   g++ -O3 -march=native -pthread -std=gnu++17 locks_asm.cpp -o locks_asm
// Run:
//   ./locks_asm
//
// Tune N/NUM_THREADS to match your experiment grid.

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>

using std::cerr;
using std::cout;
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

static inline void compiler_barrier() { __asm__ __volatile__("" ::: "memory"); }

// -------------------- Required primitives (inline asm) --------------------
static inline bool cas_primitive(bool* addr, bool expected, bool desired)
{
    unsigned char ok, exp = expected, des = desired;
    __asm__ __volatile__(
        "lock; cmpxchgb %3, %1\n\t"
        "sete %0"
        : "=q"(ok), "+m"(*addr), "+a"(exp)
        : "q"(des)
        : "memory", "cc");
    return ok;
}

static inline uint64_t fai_primitive(uint64_t* addr)  // fetch-and-increment by 1, returns old
{
    uint64_t r = 1;
    __asm__ __volatile__(
        "lock; xaddq %0, %1"
        : "+r"(r), "+m"(*addr)
        :
        : "memory", "cc");
    return r;
}

static inline uint64_t add_fetch_primitive(uint64_t* addr) // add 1, return new
{
    uint64_t r = 1;
    __asm__ __volatile__(
        "lock; xaddq %0, %1"
        : "+r"(r), "+m"(*addr)
        :
        : "memory", "cc");
    return r + 1;
}

// -------------------- Config --------------------
#define N (1000000LL)        // iterations per thread (adjust)
#define NUM_THREADS (8)      // threads

// -------------------- Shared variables --------------------
alignas(64) volatile uint64_t var1 = 0;
alignas(64) volatile uint64_t var2 = (uint64_t)N * (uint64_t)NUM_THREADS + 1;

// -------------------- Base class --------------------
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

// -------------------- Pthread mutex --------------------
class PthreadMutex : public LockBase {
private:
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
public:
    void acquire(uint16_t) override { pthread_mutex_lock(&lock); }
    void release(uint16_t) override { pthread_mutex_unlock(&lock); }
};

// -------------------- Spin (TTAS) --------------------
class SpinLock : public LockBase {
private:
    alignas(64) volatile bool flag = false;
public:
    void acquire(uint16_t) override {
        for (;;) {
            while (flag) cpu_relax(); // test
            bool expected = false;
            if (cas_primitive((bool*)&flag, expected, true)) {
                compiler_barrier(); // acquire
                break;
            }
            for (int k = 0; k < 64; ++k) cpu_relax(); // gentle backoff
        }
    }
    void release(uint16_t) override {
        compiler_barrier(); // release
        flag = false;
    }
};

// -------------------- Ticket lock --------------------
class TicketLock : public LockBase {
private:
    alignas(64) volatile uint64_t next  = 0;
    alignas(64) volatile uint64_t owner = 0;
public:
    void acquire(uint16_t) override {
        uint64_t my = fai_primitive((uint64_t*)&next); // old value
        while (owner != my) cpu_relax();
        compiler_barrier();
    }
    void release(uint16_t) override {
        (void)add_fetch_primitive((uint64_t*)&owner);
    }
};

// -------------------- Filter lock --------------------
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
            level[tid] = L;
            victim[L] = (int)tid;
            compiler_barrier();
            bool spin;
            do {
                spin = false;
                for (int k = 0; k < NUM_THREADS; ++k) {
                    if (k == tid) continue;
                    int lk = level[k];
                    int vl = victim[L];
                    if (lk >= L && vl == (int)tid) { spin = true; cpu_relax(); break; }
                }
            } while (spin);
        }
        compiler_barrier();
    }
    void release(uint16_t tid) override {
        compiler_barrier();
        level[tid] = 0;
    }
};

// -------------------- Bakery lock (Lamport) --------------------
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
        choosing[tid] = true; compiler_barrier();
        uint64_t maxn = 0;
        for (int k = 0; k < NUM_THREADS; ++k) {
            uint64_t nk = number[k];
            if (nk > maxn) maxn = nk;
        }
        number[tid] = maxn + 1; compiler_barrier();
        choosing[tid] = false; compiler_barrier();

        for (int k = 0; k < NUM_THREADS; ++k) {
            if (k == tid) continue;
            while (choosing[k]) cpu_relax();
            for (;;) {
                uint64_t nk = number[k];
                if (nk == 0) break;
                uint64_t nt = number[tid];
                if (!before(nk, k, nt, tid)) break;
                cpu_relax();
            }
        }
        compiler_barrier();
    }
    void release(uint16_t tid) override {
        compiler_barrier();
        number[tid] = 0;
    }
};

// -------------------- Anderson Array-based Queue lock --------------------
class ArrayQLock : public LockBase {
private:
    static constexpr uint32_t SZ = 1u << 6; // 64 slots (>= NUM_THREADS), power of two
    alignas(64) volatile bool flags[SZ];
    alignas(64) volatile uint64_t next = 0;
    alignas(64) volatile uint32_t slot_of[NUM_THREADS];
public:
    ArrayQLock() {
        for (uint32_t i = 0; i < SZ; ++i) flags[i] = false;
        flags[0] = true;
        for (int i = 0; i < NUM_THREADS; ++i) slot_of[i] = 0;
    }
    void acquire(uint16_t tid) override {
        uint64_t ticket = fai_primitive((uint64_t*)&next);
        uint32_t my = (uint32_t)(ticket & (SZ - 1));
        slot_of[tid] = my;
        while (!flags[my]) cpu_relax();
        flags[my] = false;
        compiler_barrier();
    }
    void release(uint16_t tid) override {
        uint32_t my = slot_of[tid];
        uint32_t nxt = (my + 1) & (SZ - 1);
        compiler_barrier();
        flags[nxt] = true;
    }
};

// -------------------- Benchmark plumbing --------------------
volatile uint64_t summed_ms = 0;

inline void critical_section() {
    // Not atomic; guarded by the lock
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

    (void)add_fetch_primitive((uint64_t*)&summed_ms);
    // summed_ms += duration;  // We want add of 'duration'. Use a CAS loop:
    uint64_t cur, nxt;
    do {
        cur = summed_ms;
        nxt = cur + duration;
        // A 64-bit CAS using cmpxchgq
        unsigned char ok;
        __asm__ __volatile__(
            "lock; cmpxchgq %3, %1\n\t"
            "sete %0"
            : "=q"(ok), "+m"(summed_ms), "+a"(cur)
            : "r"(nxt)
            : "memory", "cc");
        if (ok) break;
    } while (true);

    return nullptr;
}

static void run_one(const char* name, LockBase* lock_obj) {
    var1 = 0;
    var2 = (uint64_t)N * (uint64_t)NUM_THREADS + 1;
    summed_ms = 0;

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

    cout << name << ": summed thread time (ms): " << (unsigned long long)summed_ms << "\n";
}

int main() {
    int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
    if (error != 0) { cerr << "Error in barrier init.\n"; return 1; }

    cout << "N=" << (long long)N << ", threads=" << NUM_THREADS << "\n";

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
