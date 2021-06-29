CU=nvcc
CUFLAGS=-O2 -Xcompiler=-Wall -Xcompiler=-Wextra -arch=sm_50 -Wno-deprecated-gpu-targets
LDFLAGS=  -lcurand

TARGETS=tiny_mc_gpu.cu

all: $(TARGETS)

$(TARGETS): intermedio.o
	$(CU) $(CUFLAGS) -o $@ $^ $(LDFLAGS)

# CU, CUFLAGS no tienen regla implicita como CC/CFLAGS
%.o: %.cu
	$(CU) $(CUFLAGS) -o $@ -c $<

.PHONY: clean

clean:
	rm -f *.o $(TARGETS)

