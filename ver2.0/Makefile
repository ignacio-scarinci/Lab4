CU=nvcc
CUFLAGS=-O3 -Xcompiler=-Wall -Xcompiler=-Wextra -arch=sm_50 -Wno-deprecated-gpu-targets -use_fast_math -lcurand
LDFLAGS=

TARGETS=tiny_mc_gpu

all: $(TARGETS)

tiny_mc_gpu: tiny_mc_gpu.o
	$(CU) $(CUFLAGS) -o $@ $^ $(LDFLAGS)

# CU, CUFLAGS no tienen regla implicita como CC/CFLAGS
%.o: %.cu
	$(CU) $(CUFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	rm -f *.o $(TARGETS)
