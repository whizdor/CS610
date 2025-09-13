#include <atomic>
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
using std::chrono::microseconds;
using std::chrono::milliseconds;

// --------------------- Constants -------------------------------
#ifndef N
#define N (1e7)
#endif
#ifndef NUM_THREADS
#define NUM_THREADS 16
#endif


// -------------------- Pause ------------------------------
static inline void cpu_relax()
{
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

// --------------------- Memory Fence --------------------
static inline void compiler_barrier() 
{ 
    __asm__ __volatile__("mfence" ::: "memory"); 
}

// -------------------- Compare and Swap (CAS) --------------------
static inline bool cas_primitive(bool *addr, bool expected, bool desired)
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

// -------------------- Fetch-and-Increment (FAI) --------------------
static inline uint64_t fai_primitive(uint64_t* addr)
{
    uint64_t r = 1;
    __asm__ __volatile__(
        "lock; xaddq %0, %1"
        : "+r"(r), "+m"(*addr)
        :
        : "memory", "cc");
    return r;
}

// -------------------- Add-and-Fetch --------------------
static inline uint64_t add_fetch_primitive(uint64_t *addr) // add 1, return new
{
    uint64_t r = 1;
    __asm__ __volatile__(
        "lock; xaddq %0, %1"
        : "+r"(r), "+m"(*addr)
        :
        : "memory", "cc");
    return r + 1;
}


// Shared variables
uint64_t var1 = 0;
uint64_t var2 = (N * NUM_THREADS + 1);

constexpr size_t CLS = 128;

// Simple padded atomic wrapper to avoid false sharing for arrays.
struct alignas(64) AlignedInt { int v; };
struct alignas(64) AlignedBool { bool v; };
struct alignas(64) AlignedUint64 { uint64_t v; };


// Abstract base class
class LockBase
{
public:
    // Pure virtual function
    virtual void acquire(uint16_t tid) = 0;
    virtual void release(uint16_t tid) = 0;
};

typedef struct thr_args
{
    uint16_t m_id;
    LockBase *m_lock;
} ThreadArgs;

/** Use pthread mutex to implement lock routines */
class PthreadMutex : public LockBase
{
public:
    void acquire(uint16_t tid) override { pthread_mutex_lock(&lock); }
    void release(uint16_t tid) override { pthread_mutex_unlock(&lock); }

private:
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
};

class FilterLock : public LockBase
{
private: 
    alignas(CLS) volatile AlignedInt level[NUM_THREADS];
    alignas(CLS) volatile AlignedInt victim[NUM_THREADS];
public:
    void acquire(uint16_t tid) override 
    {   
        // Attempt to enter each level
        for(int L = 1; L < NUM_THREADS; L++)
        {
            // Visit level L
            level[tid].v = L;
            victim[L].v = (int) tid;
            compiler_barrier();
            // Spin while there is still a conflict.
            bool spin;
            do
            {
                spin = false;
                for (int k = 0; k < NUM_THREADS; k++)
                {
                    if (k == tid) continue;
                    int lk = level[k].v;
                    int vl = victim[L].v;
                    if (lk >= L && vl == (int) tid)
                    { 
                        spin = true; 
                        cpu_relax(); 
                        break;
                    }
                }
            } while (spin);
        }
        compiler_barrier();
    }

    void release(uint16_t tid) override 
    {
        compiler_barrier();
        level[tid].v = 0;
    }

    FilterLock()
    {
        for(int i = 0; i < NUM_THREADS; i++)
        {
            level[i].v = 0;
            victim[i].v = -1;
        }
    }
    ~FilterLock() {}
};

class BakeryLock : public LockBase
{
private:
    alignas(CLS) volatile AlignedBool choosing[NUM_THREADS];
    alignas(CLS) volatile AlignedUint64 number[NUM_THREADS];

    static inline bool before(uint64_t a_n, int a_id, uint64_t b_n, int b_id) 
    {
        return (a_n < b_n) || (a_n == b_n && a_id < b_id);
    }
public:
    void acquire(uint16_t tid) override 
    {
        choosing[tid].v = true;
        compiler_barrier();
        uint64_t max = 0;
        for (int k = 0; k < NUM_THREADS; k++)
        {
            uint64_t nk = number[k].v;
            if (nk > max) max = nk;
        }
        number[tid].v = max + 1;
        compiler_barrier();
        choosing[tid].v = false;
        compiler_barrier();

        for (int k = 0; k < NUM_THREADS; k++)
        {
            if (k == tid) continue;
            while (choosing[k].v) cpu_relax();
            for (;;)
            {
                uint64_t nk = number[k].v;
                if (nk == 0) break;
                uint64_t nt = number[tid].v;
                if (!before(nk, k, nt, tid)) break;
                cpu_relax();
            }
        }
        compiler_barrier();
    }
    void release(uint16_t tid) override 
    {
        compiler_barrier();
        number[tid].v = 0;
    }

    BakeryLock() 
    {
        for (int i = 0; i < NUM_THREADS; i++) 
        { 
            choosing[i].v = false; 
            number[i].v = 0; 
        }
    }
    ~BakeryLock() {}
};

class SpinLock : public LockBase
{
private:
    alignas(CLS) volatile bool flag = false;

public:
    void acquire(uint16_t) override
    {
        for (;;)
        {
            while (flag)
                cpu_relax();
            bool expected = false;
            if (cas_primitive((bool *)&flag, expected, true))
            {
                compiler_barrier(); // fence for acquire
                break;
            }
            for (int k = 0; k < 64; ++k)
                cpu_relax(); // backoff
        }
    }
    void release(uint16_t) override
    {
        compiler_barrier(); // fence for release
        flag = false;
    }

    SpinLock() {}
    ~SpinLock() {}
};

class TicketLock : public LockBase
{
private:
    alignas(CLS) volatile uint64_t next = 0;
    alignas(CLS) volatile uint64_t owner = 0;
public:
    void acquire(uint16_t tid) override
    {
        uint64_t my = fai_primitive((uint64_t *)&next);
        while (owner != my)
            cpu_relax();
        compiler_barrier();
    }
    void release(uint16_t tid) override
    {
        (void)add_fetch_primitive((uint64_t *)&owner);
    }

    TicketLock() {}
    ~TicketLock() {}
};

class ArrayQLock : public LockBase
{
private:
    static constexpr uint32_t SZ = NUM_THREADS;
    alignas(CLS) volatile AlignedBool flags[SZ];
    alignas(CLS) volatile AlignedUint64 next;
    // Per-thread slot index
    alignas(CLS) volatile AlignedUint64 slot[NUM_THREADS];
public:
    void acquire(uint16_t tid) override 
    {
        uint64_t ticket = fai_primitive((uint64_t*)&next);
        uint32_t my = (uint32_t)(ticket & (SZ - 1));
        slot[tid].v = my;
        while (!flags[my].v) 
            cpu_relax();
        flags[my].v = false;
        compiler_barrier();
    }
    void release(uint16_t tid) override 
    {
        uint32_t my = slot[tid].v;
        uint32_t nxt = (my + 1) & (SZ - 1);
        compiler_barrier();
        flags[nxt].v = true;
    }

    ArrayQLock() 
    {
        for(uint32_t i = 0; i < SZ; ++i) 
            flags[i].v = false;
        flags[0].v = true;
        for (int i = 0; i < NUM_THREADS; i++) 
            slot[i].v = 0;
    }
    ~ArrayQLock() {}
};

/** Estimate the time taken */
std::atomic_uint64_t sync_time = 0;

inline void critical_section()
{
    var1++;
    var2--;
}

/** Sync threads at the start to maximize contention */
pthread_barrier_t g_barrier;

void *thrBody(void *arguments)
{
    ThreadArgs *tmp = static_cast<ThreadArgs *>(arguments);
    if (false)
    {
        cout << "Thread id: " << tmp->m_id << " starting\n";
    }

    // Wait for all other producer threads to launch before proceeding.
    pthread_barrier_wait(&g_barrier);

    HRTimer start = HR::now();
    for (int i = 0; i < N; i++)
    {
        tmp->m_lock->acquire(tmp->m_id);
        critical_section();
        tmp->m_lock->release(tmp->m_id);
    }
    HRTimer end = HR::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    // A barrier is not required here
    sync_time.fetch_add(duration);
    pthread_exit(NULL);
}

int main()
{
    int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
    if (error != 0)
    {
        cerr << "Error in barrier init.\n";
        exit(EXIT_FAILURE);
    }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_t tid[NUM_THREADS];
    ThreadArgs args[NUM_THREADS] = {{0}};

    // Pthread mutex
    LockBase *lock_obj = new PthreadMutex();
    uint16_t i = 0;
    while (i < NUM_THREADS)
    {
        args[i].m_id = i;
        args[i].m_lock = lock_obj;

        error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
        if (error != 0)
        {
            cerr << "\nThread cannot be created : " << strerror(error) << "\n";
            exit(EXIT_FAILURE);
        }
        i++;
    }

    i = 0;
    void *status;
    while (i < NUM_THREADS)
    {
        error = pthread_join(tid[i], &status);
        if (error)
        {
            cerr << "ERROR: return code from pthread_join() is " << error << "\n";
            exit(EXIT_FAILURE);
        }
        i++;
    }

    assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    cout << "Pthread mutex: Time taken (us): " << sync_time << "\n";

    // Filter lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time.store(0);

    lock_obj = new FilterLock();
    i = 0;
    while (i < NUM_THREADS)
    {
        args[i].m_id = i;
        args[i].m_lock = lock_obj;

        error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
        if (error != 0)
        {
            printf("\nThread cannot be created : [%s]", strerror(error));
            exit(EXIT_FAILURE);
        }
        i++;
    }

    i = 0;
    while (i < NUM_THREADS)
    {
        error = pthread_join(tid[i], &status);
        if (error)
        {
            printf("ERROR: return code from pthread_join() is %d\n", error);
            exit(EXIT_FAILURE);
        }
        i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    // assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Filter lock: Time taken (us): " << sync_time << "\n";

    // Bakery lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time.store(0);

    lock_obj = new BakeryLock();
    i = 0;
    while (i < NUM_THREADS)
    {
        args[i].m_id = i;
        args[i].m_lock = lock_obj;

        error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
        if (error != 0)
        {
            printf("\nThread cannot be created : [%s]", strerror(error));
            exit(EXIT_FAILURE);
        }
        i++;
    }

    i = 0;
    while (i < NUM_THREADS)
    {
        error = pthread_join(tid[i], &status);
        if (error)
        {
            printf("ERROR: return code from pthread_join() is %d\n", error);
            exit(EXIT_FAILURE);
        }
        i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    // assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Bakery lock: Time taken (us): " << sync_time << "\n";

    // Spin lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time.store(0);

    lock_obj = new SpinLock();
    i = 0;
    while (i < NUM_THREADS)
    {
        args[i].m_id = i;
        args[i].m_lock = lock_obj;

        error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
        if (error != 0)
        {
            printf("\nThread cannot be created : [%s]", strerror(error));
            exit(EXIT_FAILURE);
        }
        i++;
    }

    i = 0;
    while (i < NUM_THREADS)
    {
        error = pthread_join(tid[i], &status);
        if (error)
        {
            printf("ERROR: return code from pthread_join() is %d\n", error);
            exit(EXIT_FAILURE);
        }
        i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Spin lock: Time taken (us): " << sync_time << "\n";

    // Ticket lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time.store(0);

    lock_obj = new TicketLock();
    i = 0;
    while (i < NUM_THREADS)
    {
        args[i].m_id = i;
        args[i].m_lock = lock_obj;

        error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
        if (error != 0)
        {
            printf("\nThread cannot be created : [%s]", strerror(error));
            exit(EXIT_FAILURE);
        }
        i++;
    }

    i = 0;
    while (i < NUM_THREADS)
    {
        error = pthread_join(tid[i], &status);
        if (error)
        {
            printf("ERROR: return code from pthread_join() is %d\n", error);
            exit(EXIT_FAILURE);
        }
        i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Ticket lock: Time taken (us): " << sync_time << "\n";

    // Array Q lock
    var1 = 0;
    var2 = (N * NUM_THREADS + 1);
    sync_time.store(0);

    lock_obj = new ArrayQLock();
    i = 0;
    while (i < NUM_THREADS)
    {
        args[i].m_id = i;
        args[i].m_lock = lock_obj;

        error = pthread_create(&tid[i], &attr, thrBody, (void *)(args + i));
        if (error != 0)
        {
            printf("\nThread cannot be created : [%s]", strerror(error));
            exit(EXIT_FAILURE);
        }
        i++;
    }

    i = 0;
    while (i < NUM_THREADS)
    {
        error = pthread_join(tid[i], &status);
        if (error)
        {
            printf("ERROR: return code from pthread_join() is %d\n", error);
            exit(EXIT_FAILURE);
        }
        i++;
    }

    cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
    // assert(var1 == N * NUM_THREADS && var2 == 1);
    cout << "Array Q lock: Time taken (us): " << sync_time << "\n";

    pthread_barrier_destroy(&g_barrier);
    pthread_attr_destroy(&attr);

    pthread_exit(NULL);
}
