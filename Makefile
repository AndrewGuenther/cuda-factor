CC=g++
NVCC=nvcc
CFLAGS=-g
CUDA_FLAGS=--relocatable-device-code true -g -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
DEBUG=-g
LIBS=-lgmp

all: cpu_crack gpu_crack

cpu_crack: cpu_crack.c utils.o
	$(CC) $(CFLAGS) -o cpu_crack cpu_crack.c utils.o $(LIBS)

gpu_crack: gpu_crack.cu cuda_utils.o utils.o
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) -o gpu_crack gpu_crack.cu cuda_utils.o utils.o $(LIBS)

test: test_cuda.cu cuda_utils.o utils.o
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) -o test_cuda test_cuda.cu cuda_utils.o utils.o $(LIBS)

cuda_utils.o: cuda_utils.cu cuda_utils.h
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) cuda_utils.cu $(LIBS) -c

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) utils.c $(LIBS) -c

clean:
	rm cpu_crack gpu_crack test_cuda cuda_utils.o utils.o
