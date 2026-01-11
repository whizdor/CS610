#include <cstddef>
#include <chrono>

const size_t SIZE = 256 * 1024 * 1024; // 256 MB
char *dummy = new char[SIZE];

void flush_cache() 
{
    for (size_t i = 0; i < SIZE; i++) 
    { 
        dummy[i] += 1;
    }
}

int main()
{
    flush_cache();
    return 0;
}