CC=gcc
CXX=g++

CFLAGS=-O3 -std=c17 -mavx -mavx2
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
# Preprocessing flags, not CPP flags
CPPFLAGS=

LIBS=-pthread

# First target is the default target
all: problem1 problem2 problem3

BIN=./bin
$(shell mkdir -p $(BIN))

.DEFAULT_TARGET: all

# $@ is a variable that contains the target name.
problem1: ./problem1-dir/220573-prob1.cpp
	g++ -O3 -march=native -std=c++17 -DUSE_PAPI -mavx -mavx2 ./problem1-dir/220573-prob1.cpp -lpapi -o ./bin/problem1.out

problem2: ./problem2-dir/220573-prob2.cpp
	g++ -std=c++20 problem2-dir/220573-prob2.cpp -o ./bin/problem2.out

problem2_efficient: ./problem2-dir/220573-prob2_modified.cpp
	g++ -std=c++20 problem2-dir/220573-prob2_modified.cpp -o ./bin/problem2_efficient.out

problem3: ./problem3-dir/220573-prob3.cpp
	g++ -std=c++20 problem3-dir/220573-prob3.cpp -o ./bin/problem3.out

# The target is not a filename, always run the rule even if there is a file
.PHONY: clean
clean:
	rm -f -- bin *.out compile_commands.json