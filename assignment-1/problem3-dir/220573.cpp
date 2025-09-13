#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <random>
using namespace std;

// Helper Functions
int min(int a, int b) { return (a < b) ? a : b; }

struct Counter
{
    mutex counter_lock;
    mutex producer_lock;

    int active_producers = 0;
    int active_consumers = 0;

    Counter()
    {
        active_producers = 0;
        active_consumers = 0;
    }
};

struct LineBuffer
{
    queue<vector<string>> q;

    int lines_in_buf = 0;
    int capacity_lines;
    bool producers_done = false;

    mutex buffer_lock;
    condition_variable cv_has_data;
    condition_variable cv_has_space;

    LineBuffer(int capacity)
    {
        capacity_lines = capacity;
        producers_done = false;
    }
};

struct InputFile
{
    ifstream in;
    mutex input_file_lock;

    InputFile(const string &path) : in(path)
    {
        if (!in.is_open())
        {
            cerr << "[ERROR]: Failed to open input file: " << path << endl;
            exit(1);
        }
    }

    bool read_lines(int L, vector<string> &lines)
    {
        lines.clear();
        lines.reserve(L);
        unique_lock<mutex> lk(input_file_lock);
        for (int i = 0; i < L; ++i)
        {
            string line;
            if (!getline(in, line))
            {
                if (lines.empty())
                {
                    lk.unlock();
                    return false; // EOF reached
                }
                break; // EOF reached, but we have some lines
            }
            lines.push_back(std::move(line));
        }
        lk.unlock();
        return true;
    }
};

struct OutputFile
{
    ofstream out;
    mutex output_file_lock;

    OutputFile(const string &path) : out(path)
    {
        if (!out.is_open())
        {
            cerr << "[ERROR]: Failed to open output file: " << path << endl;
            exit(1);
        }
    }

    void write_lines(const vector<string> &lines)
    {
        unique_lock<mutex> lk(output_file_lock);
        for (const auto &line : lines)
        {
            out << line << '\n';
        }
        out.flush();
        lk.unlock();
    }
};

void producer(LineBuffer &buf, InputFile &file, Counter &counter, int T, int Lmin, int Lmax, int M)
{
    // The number of lines read by producer is calculated here.
    // Randomly choose how many lines to read
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(Lmin, Lmax);
    int L = dis(gen);

    while (true)
    {
        if (L < 1)
        {
            continue;
        }

        vector<string> lines;
        if (!file.read_lines(L, lines))
        {
            // EOF Reached
            break;
        }
        // Enqueue these lines atomically with respect to other producers.
        unique_lock<mutex> turn(counter.producer_lock);

        int idx = 0;
        while (idx < lines.size())
        {
            int want = min(M, lines.size() - idx);
            // Wait for enough space for 'want' lines
            unique_lock<mutex> lk(buf.buffer_lock);
            buf.cv_has_space.wait(lk, [&]
                                  { return (buf.lines_in_buf + want <= buf.capacity_lines); });

            vector<string> chunk;
            chunk.reserve(want);
            for (int i = 0; i < want; i++)
            {
                chunk.push_back(std::move(lines[idx + i]));
            }

            buf.q.push(std::move(chunk));
            buf.lines_in_buf += want;
            buf.cv_has_data.notify_one();

            lk.unlock();
            idx += want;
        }
        turn.unlock();
    }

    unique_lock<mutex> g(counter.counter_lock);
    counter.active_producers--;
    bool i_am_last = (counter.active_producers == 0);
    g.unlock();

    if (i_am_last)
    {
        unique_lock<mutex> lk(buf.buffer_lock);
        buf.producers_done = true;
        buf.cv_has_data.notify_all();
        lk.unlock();
    }
}

void consumer(LineBuffer &buf, OutputFile &file)
{
    while (true)
    {
        vector<string> lines;
        {
            unique_lock<mutex> lk(buf.buffer_lock);

            buf.cv_has_data.wait(lk, [&]
                                 { return !buf.q.empty() || buf.producers_done; });

            if (buf.q.empty() && buf.producers_done)
            {
                break; // No more data to consume
            }

            lines = std::move(buf.q.front());
            buf.q.pop();
            buf.lines_in_buf -= lines.size();
            file.write_lines(lines);
            buf.cv_has_space.notify_all(); // Notify producers that space is available
            lk.unlock();
        }
    }
}

signed main(int argc, char **argv)
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    if (argc != 7)
    {
        cout << "[ERROR]: Usage: " << argv[0] << " <input_file> <T> <Lmin> <Lmax> <M> <output_file>" << endl;
        return 1;
    }

    string R = argv[1];
    int T = stoi(argv[2]);
    int Lmin = stoi(argv[3]);
    int Lmax = stoi(argv[4]);
    int M = stoi(argv[5]);
    string W = argv[6];

    if (T <= 0 || Lmin <= 0 || Lmax < Lmin || M <= 0)
    {
        cout << "[ERROR]: Invalid arguments.\n";
        return 2;
    }

    ifstream in(R);
    if (!in)
    {
        cout << "[ERROR]: Failed to open input: " << R << "\n";
        return 3;
    }
    ofstream out(W, ios::out | ios::trunc);
    if (!out)
    {
        cout << "[ERROR]: Failed to open output: " << W << "\n";
        return 4;
    }

    LineBuffer buf(M);
    OutputFile output(W);
    Counter counter;
    InputFile input(R);

    vector<thread> producers;
    for (int i = 0; i < T; i++)
    {
        counter.active_producers++;
        producers.emplace_back(producer, ref(buf), ref(input), ref(counter), T, Lmin, Lmax, M);
    }

    vector<thread> consumers;
    for (int i = 0; i < T; i++)
    {
        consumers.emplace_back(consumer, ref(buf), ref(output));
    }

    cout << "[MAIN] All producers and consumers have spawned.\n";

    for (auto &p : producers)
        p.join();
    for (auto &c : consumers)
        c.join();

    cout << "[MAIN] All producers and consumers have finished.\n";

    return 0;
}