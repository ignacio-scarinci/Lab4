#!/bin/bash

for ((dim_grid=10; dim_grid<101; dim_grid+=10));do
	for dim_block in 32 64 192 256 320
	do
	./tiny_mc_gpu 10000000 ${dim_grid} ${dim_block} >> ./res_exp1/results_${dim_grid}_${dim_block}.dat
	done
done
