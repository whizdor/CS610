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

#define N (1e7)
#define NUM_THREADS (8)

uint64_t var1 = 0, var2 = (N * NUM_THREADS + 1);

constexpr std::size_t CLS = 128;
constexpr std::size_t STRIDE = (CLS + sizeof(uint64_t) - 1) / sizeof(uint64_t);


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

static inline uint64_t add_fetch_primitive(uint64_t* addr)
{
    uint64_t r = 1;
    __asm__ __volatile__(
        "lock; xaddq %0, %1"
        : "+r"(r), "+m"(*addr)
        :
        : "memory", "cc");
    return r + 1;
}

class LockBase 
{
  public:
    virtual void acquire(uint16_t tid) = 0;
    virtual void release(uint16_t tid) = 0;
};

typedef struct thr_args 
{
  uint16_t m_id;
  LockBase* m_lock;
} ThreadArgs;

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
    alignas(CLS) uint64_t level[NUM_THREADS * STRIDE];
    uint64_t victim[NUM_THREADS];

  public:

    void acquire(uint16_t tid) override 
    {
      for (uint64_t i = 1; i < NUM_THREADS; i++) 
      {
        level[tid * STRIDE] = i;
        victim[i] = tid;

        std::atomic_thread_fence(std::memory_order_seq_cst);

        while(otherThreadsOnSameOrHigherLevel(tid, i) && (victim[i] == tid)) ;
      }
    }

    void release(uint16_t tid) override 
    {
      level[tid * STRIDE] = 0;
    }

    bool otherThreadsOnSameOrHigherLevel(uint16_t tid, uint64_t myLevel) 
    {
      for (uint64_t i = 0; i < NUM_THREADS; i++) 
      {
        if ((i != tid) && (level[i * STRIDE] >= myLevel)) 
        {
          return true;
        }
      }
      return false;
    }

    FilterLock() 
    {
      for (int i = 0; i < NUM_THREADS; i++) 
      {
        level[i * STRIDE] = 0;
      }
    }
    ~FilterLock() 
    {
    }
};

class BakeryLock : public LockBase 
{
  private:
    uint64_t max;
    bool choosing[NUM_THREADS];
    uint64_t number[NUM_THREADS];
  public:
    void acquire(uint16_t tid) override 
    {
      choosing[tid] = true;
      number[tid] = add_fetch_primitive(&max);
      
      std::atomic_thread_fence(std::memory_order_seq_cst);

      while(otherThreadsChoosingOrHigherLabel(tid)) ;
    }

    bool otherThreadsChoosingOrHigherLabel(uint16_t tid) 
    {
      for (int i = 0; i < NUM_THREADS; i++) 
      {
        if (i != tid && (choosing[i] && number[i] < number[tid])) 
        {
          return true;
        }
      }
      return false;
    }

    void release(uint16_t tid) override 
    {
      choosing[tid] = false;
    }

    BakeryLock() 
    {
      max = 0;
      for (int i = 0; i<NUM_THREADS; i++)
      {
        choosing[i] = false;
        number[i] = 0;
      }
    }
    ~BakeryLock() {}
};

class SpinLock : public LockBase 
{
  private:
    bool locked;
  public:
    void acquire(uint16_t tid) override 
    {
      while (!cas_primitive(&locked, false, true)) ;
    }
    void release(uint16_t tid) override 
    {
      locked = false;
    }

    SpinLock() 
    {
      locked = false;
    }
    ~SpinLock() {}
};

class TicketLock : public LockBase 
{
  private:
    uint64_t next_ticket = 0;
    uint64_t serving_ticket = 0;
  public:
    void acquire(uint16_t tid) override 
    {
      uint64_t my_ticket = fai_primitive(&next_ticket);
      while (serving_ticket != my_ticket) ;
    }
    void release(uint16_t tid) override 
    {
      serving_ticket++;
    }

    TicketLock() {}
    ~TicketLock() {}
};

class ArrayQLock : public LockBase 
{
  private:
  uint64_t tail;
  alignas(CLS) uint64_t slot[NUM_THREADS * STRIDE];
  bool flag[NUM_THREADS];

  public:
    void acquire(uint16_t tid) override 
    {
      uint64_t my_slot = fai_primitive(&tail) % NUM_THREADS;
      slot[tid * STRIDE] = my_slot;

      std::atomic_thread_fence(std::memory_order_seq_cst);

      while (!flag[my_slot]) ;
    }
    void release(uint16_t tid) override 
    {
      uint64_t my_slot = slot[tid * STRIDE];
      flag[my_slot] = false;
      flag[(my_slot + 1) % NUM_THREADS] = true;
    }

    ArrayQLock() 
    {
      tail = 0;
      for (int i = 0; i < NUM_THREADS; i++)  
      {
        flag[i] = false;
      }
      flag[0] = true;
    }
    ~ArrayQLock() {}
};

std::atomic_uint64_t sync_time = 0;

inline void critical_section() {
  var1++;
  var2--;
}

/** Sync threads at the start to maximize contention */
pthread_barrier_t g_barrier;

void* thrBody(void* arguments) {
  ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);
  if (false) {
    cout << "Thread id: " << tmp->m_id << " starting\n";
  }

  // Wait for all other producer threads to launch before proceeding.
  pthread_barrier_wait(&g_barrier);

  HRTimer start = HR::now();
  for (int i = 0; i < N; i++) {
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
  LockBase* lock_obj = new PthreadMutex();
  uint16_t i = 0;
  while (i < NUM_THREADS) 
  {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) 
    {
      cerr << "\nThread cannot be created : " << strerror(error) << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  void* status;
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

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
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

  assert(var1 == N * NUM_THREADS && var2 == 1);
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

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
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

  assert(var1 == N * NUM_THREADS && var2 == 1);
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

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
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

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
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

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
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

  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Array Q lock: Time taken (us): " << sync_time << "\n";

  pthread_barrier_destroy(&g_barrier);
  pthread_attr_destroy(&attr);

  pthread_exit(NULL);
}
