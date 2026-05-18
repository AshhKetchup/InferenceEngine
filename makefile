CC = gcc
CXX = g++
CFLAGS = -O2 -Wall
CXXFLAGS = -O2 -std=c++17 -Wall

infer-eng: infer-eng.o gguflib.o fp16.o
	$(CXX) $^ -o $@

infer-eng.o: infer-eng.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

gguflib.o: include/src/gguflib.c
	$(CC) $(CFLAGS) -c $< -o $@

fp16.o: include/src/fp16.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o infer-eng
