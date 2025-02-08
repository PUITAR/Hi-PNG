CXX = g++
CXXFLAGS = -std=c++17 -fopenmp -Ofast -march=native -mtune=native -lrt \
  -I../anns/include \
  -I../interval

TARGET = groundtruth
SOURCES = groundtruth.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)