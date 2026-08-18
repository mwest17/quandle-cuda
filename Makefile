CXX := g++
NVCC := nvcc

CPPFLAGS := -Iincl
CXXFLAGS := -std=c++17 -O2
NVCCFLAGS := -std=c++17 -O3 --compiler-bindir "C:\msys64\ucrt64\bin"

TARGET := quandle_cuda
SRC := src/main.cpp
OBJ := main.o

.PHONY: all clean run

all: $(TARGET)

$(OBJ): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $^ -o $@

run: $(TARGET)
	$(TARGET)

clean:
	@if exist $(OBJ) del $(OBJ)
	@if exist $(TARGET).exe del $(TARGET).exe
	@if exist $(TARGET) del $(TARGET)
