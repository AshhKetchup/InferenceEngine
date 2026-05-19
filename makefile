CC      = gcc
CXX     = g++
CFLAGS  = -O2 -Wall
CXXFLAGS = -O2 -std=c++17 -Wall

OBJS = infer-eng.o src/tokenizer.o src/config.o gguflib.o fp16.o

infer-eng: $(OBJS)
	$(CXX) $^ -o $@

infer-eng.o: infer-eng.cpp src/gguf_model.h src/tokenizer.h src/config.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/tokenizer.o: src/tokenizer.cpp src/tokenizer.h src/gguf_model.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/config.o: src/config.cpp src/config.h src/tokenizer.h src/gguf_model.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

gguflib.o: include/src/gguflib.c
	$(CC) $(CFLAGS) -c $< -o $@

fp16.o: include/src/fp16.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o src/*.o infer-eng
