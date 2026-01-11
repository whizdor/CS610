#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::ios;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

constexpr std::size_t CLS = 64; // assume 64B cache line

struct alignas(CLS) PaddedCounter {
    uint64_t value = 0;
};

struct t_data {
    uint32_t tid;
};

struct word_tracker {
    std::vector<PaddedCounter> word_count; // dynamically sized, padded
    uint64_t total_lines_processed;
    uint64_t total_words_processed;
    pthread_mutex_t totals_mutex;
} tracker;

std::queue<std::string> shared_pq;
pthread_mutex_t pq_mutex = PTHREAD_MUTEX_INITIALIZER;

int MAX_THREADS = 0;

void* thread_runner(void*);

void print_usage(const char* prog_name) {
    cerr << "usage: " << prog_name << " <thread count> <input file>\n";
    exit(EXIT_FAILURE);
}

void print_counters() {
    for (int id = 0; id < MAX_THREADS; ++id) {
        cout << "Thread " << id << " counter: "
             << tracker.word_count[id].value << '\n';
    }
}

void fill_producer_buffer(const std::string& input) {
    std::ifstream input_file(input);
    if (!input_file.is_open()) {
        cerr << "Error opening the top-level input file!\n";
        exit(EXIT_FAILURE);
    }

    std::filesystem::path p(input);
    std::string line;
    while (std::getline(input_file, line)) {
        shared_pq.push((p.parent_path() / line).string());
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
    }

    MAX_THREADS = std::strtol(argv[1], nullptr, 10);
    std::string input = argv[2];
    fill_producer_buffer(input);

    tracker.word_count.resize(MAX_THREADS);
    tracker.total_lines_processed = 0;
    tracker.total_words_processed = 0;
    tracker.totals_mutex = PTHREAD_MUTEX_INITIALIZER;

    std::vector<pthread_t> threads(MAX_THREADS);
    std::vector<t_data> args(MAX_THREADS);

    HRTimer start_time = HR::now();

    for (int i = 0; i < MAX_THREADS; i++) {
        args[i].tid = i;
        pthread_create(&threads[i], nullptr, thread_runner, &args[i]);
    }
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    HRTimer end_time = HR::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time).count();
    cout << "Time taken (ms): " << duration << "\n";

    print_counters();
    cout << "Total words processed: " << tracker.total_words_processed << "\n";
    cout << "Total lines processed: " << tracker.total_lines_processed << "\n";

    return EXIT_SUCCESS;
}

void* thread_runner(void* th_args) {
    auto* args = static_cast<t_data*>(th_args);
    uint32_t thread_id = args->tid;

    while (true) {
        std::string fileName;

        pthread_mutex_lock(&pq_mutex);
        if (shared_pq.empty()) {
            pthread_mutex_unlock(&pq_mutex);
            break;
        }
        fileName = shared_pq.front();
        shared_pq.pop();
        pthread_mutex_unlock(&pq_mutex);

        std::ifstream input_file(fileName);
        if (!input_file.is_open()) {
            cerr << "Error opening input file: " << fileName << endl;
            continue;
        }

        uint64_t local_lines = 0;
        uint64_t local_words = 0;

        std::string line, tok;
        while (std::getline(input_file, line)) {
            ++local_lines;
            std::istringstream iss(line);
            while (iss >> tok) ++local_words;
        }

        tracker.word_count[thread_id].value += local_words;

        pthread_mutex_lock(&tracker.totals_mutex);
        tracker.total_lines_processed += local_lines;
        tracker.total_words_processed += local_words;
        pthread_mutex_unlock(&tracker.totals_mutex);
    }

    return nullptr;
}
