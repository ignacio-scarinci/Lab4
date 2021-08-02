#!/bin/bash

for part in 1000000 5000000 10000000 50000000 100000000
	do
	./tiny_mc_gpu ${part} 100 64  >> ./res_exp2/results_${part}.dat
done
