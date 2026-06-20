CC      = gcc
CXX     = g++
CFLAGS  = -O2 -Wall
CXXFLAGS = -O2 -std=c++17 -Wall

OBJS = infer-eng.o src/tokenizer.o src/config.o src/maths_op.o gguflib.o fp16.o

infer-eng: $(OBJS)
	$(CXX) $^ -o $@

infer-eng.o: infer-eng.cpp src/gguf_model.h src/tokenizer.h src/config.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/tokenizer.o: src/tokenizer.cpp src/tokenizer.h src/gguf_model.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/config.o: src/config.cpp src/config.h src/tokenizer.h src/gguf_model.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/maths_op.o: src/maths_op.cpp src/maths_op.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

gguflib.o: include/src/gguflib.c
	$(CC) $(CFLAGS) -c $< -o $@

fp16.o: include/src/fp16.c
	$(CC) $(CFLAGS) -c $< -o $@

TEST_OBJS = tests/run_tests.o tests/test_maths_op.o src/maths_op.o

run_tests: $(TEST_OBJS)
	$(CXX) $^ -o $@

tests/run_tests.o: tests/run_tests.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

tests/test_maths_op.o: tests/test_maths_op.cpp src/maths_op.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: run_tests
	./run_tests

clean:
	rm -f *.o src/*.o tests/*.o infer-eng run_tests
