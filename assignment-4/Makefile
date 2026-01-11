CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CXXFLAGS=-std=c++17 -masm=att -msse4 -mavx2 -march=native -fopenmp -fverbose-asm -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti -fcf-protection=none
CPPFLAGS=
CUDAFLAGS=-O2 -std=c++17 -arch=sm_80 -lineinfo -res-usage -src-in-ptx

LIBS=-pthread

# First target is the default target
all: problem1 problem2 problem3-v0 problem4

# $@ is a variable that contains the target name.
problem1%: problem1-dir/220573-problem1%.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem2: problem2-dir/220573-problem2.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-v0: problem3-dir/220573-problem3-v0.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O3 $< -o $(patsubst %.cpp,%.out,$<)

problem3-%: problem3-dir/220573-problem3-%.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem4-v%: problem4-dir/220573-problem4-v%.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

# The target is not a filename
clean:
	rm *.out compile_commands.json
