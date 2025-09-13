// Coarse-grained locking implies 1 lock for the whole map
// Fine-grained locking implies 1 lock for each key in the map, which is encouraged

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <pthread.h>
#include <queue>
#include <string>
#include <unistd.h>
#include <new>
#include <vector>
#include <chrono>

using std::cerr;
using std::cout;
using std::endl;
using std::ios;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

constexpr std::size_t CLS = 128;

const int MAX_FILES = 10;
const int MAX_SIZE = 10;
int MAX_THREADS = 5;

struct t_data { uint32_t tid; };

// Each element is over-aligned so it occupies its own cache line.
struct alignas(CLS) PaddedCounter { uint64_t value; };

struct word_tracker
{
  // allocated at runtime once thread_count is known
  std::vector<PaddedCounter> word_count;
  uint64_t total_lines_processed;
  uint64_t total_words_processed;
  pthread_mutex_t word_count_mutex;
} tracker;

std::queue<std::string> shared_pq;
pthread_mutex_t pq_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t line_count_mutex = PTHREAD_MUTEX_INITIALIZER;

void* thread_runner(void*);
void print_usage(char* prog_name)
{
  cerr << "usage: " << prog_name << " <producer count> <input file>\n";
  exit(EXIT_FAILURE);
}

void print_counters()
{
  for (int id = 0; id < MAX_THREADS; ++id)
  {
    std::cout << "Thread " << id << " counter: " << tracker.word_count[id].value << '\n';
  }
}

void fill_producer_buffer(std::string& input)
{
  std::fstream input_file;
  input_file.open(input, ios::in);
  if (!input_file.is_open())
  {
    cerr << "Error opening the top-level input file!" << endl;
    exit(EXIT_FAILURE);
  }

  std::filesystem::path p(input);
  std::string line;
  while (getline(input_file, line))
  {
    shared_pq.push((p.parent_path() / line).string());
  }
}

int thread_count = 0;

int main(int argc, char* argv[])
{
  if (argc != 3) { print_usage(argv[0]); }

  thread_count = strtol(argv[1], NULL, 10);
  MAX_THREADS = thread_count;
  std::string input = argv[2];
  fill_producer_buffer(input);

  pthread_t threads_worker[thread_count];
  auto* args_array = (t_data*)malloc(sizeof(t_data) * thread_count);

  // Allocate and zero the aligned per-thread counters.
  tracker.word_count.resize(thread_count);
  for (int i = 0; i < thread_count; ++i) tracker.word_count[i].value = 0;

  tracker.total_lines_processed = 0;
  tracker.total_words_processed = 0;
  tracker.word_count_mutex = PTHREAD_MUTEX_INITIALIZER;

  HRTimer start_time = HR::now();

  for (int i = 0; i < thread_count; i++)
  {
    args_array[i].tid = i;
    pthread_create(&threads_worker[i], nullptr, thread_runner, (void*)&args_array[i]);
  }

  for (int i = 0; i < thread_count; i++) pthread_join(threads_worker[i], NULL);

  HRTimer end_time = HR::now();
  auto duration = duration_cast<milliseconds>(end_time - start_time).count();
  cout << "Time taken (ms): " << duration << "\n";

  print_counters();
  cout << "Total words processed: " << tracker.total_words_processed << "\n";
  cout << "Total line processed: " << tracker.total_lines_processed << "\n";

  free(args_array);
  return EXIT_SUCCESS;
}

void* thread_runner(void* th_args)
{
  auto* args = (t_data*)th_args;
  uint32_t thread_id = args->tid;
  std::fstream input_file;
  std::string fileName;
  std::string line;

  pthread_mutex_lock(&pq_mutex);
  if (!shared_pq.empty()) {
    fileName = shared_pq.front();
    shared_pq.pop();
  }
  pthread_mutex_unlock(&pq_mutex);

  if (fileName.empty())
    pthread_exit(nullptr);

  input_file.open(fileName.c_str(), ios::in);
  if (!input_file.is_open())
  {
    cerr << "Error opening input file from a thread!" << endl;
    exit(EXIT_FAILURE);
  }

  uint64_t line_count = 0;
  uint64_t word_count = 0;

  while (getline(input_file, line))
  {
    ++line_count;
    // Simple space-delimited tokenization.
    size_t pos = 0;
    const std::string delim = " ";
    while ((pos = line.find(delim)) != std::string::npos)
    {
      // token = line.substr(0, pos); // not used
      ++word_count;
      line.erase(0, pos + delim.length());
    }
    // Count the trailing token if any (non-empty remainder).
    if (!line.empty()) ++word_count;
  }

  // Per-thread, cache-line isolated counter:
  tracker.word_count[thread_id].value += word_count;

  pthread_mutex_lock(&line_count_mutex);
  tracker.total_lines_processed += line_count;
  pthread_mutex_unlock(&line_count_mutex);

  pthread_mutex_lock(&tracker.word_count_mutex);
  tracker.total_words_processed += word_count;
  pthread_mutex_unlock(&tracker.word_count_mutex);

  input_file.close();
  pthread_exit(nullptr);
}
