#!/bin/bash

ALGORITMO="arvoreReg.py"
MAX_DEPTH=("" "Default" "9" "18" "37")
MIN_SAMPLE_SPLIT=("" "Default" "4" "8" "16")

for i in $(seq 10)
do
for j in $(seq 4)
do
for k in $(seq 4)
do
	EXECUCAOPERF="sudo time perf stat -x | -e /power/energy-pkg/,/power/energy-cores/,/power/energy-gpu/"
	COMANDO2="sudo perf stat -x | -e instructions,cycles,cpu-clock,cpu-migrations,branches,branch-misses,context-switches,bus-cycles,cache-references,cache-misses,mem-loads,mem-stores,L1-dcache-stores"
	APP="python3.7 $ALGORITMO ${MAX_DEPTH[j]} ${MIN_SAMPLE_SPLIT[k]}"
        $EXECUCAOPERF $COMANDO2 $APP &>Resultado_${MAX_DEPTH[j]}DEPTH${MIN_SAMPLE_SPLIT[k]}MIN_SAMPLE$[i].txt
if [ -d "max_depth${MAX_DEPTH[j]}_min_sample_split${MIN_SAMPLE_SPLIT[k]}" ]
then
	mv Resultado_${MAX_DEPTH[j]}DEPTH${MIN_SAMPLE_SPLIT[k]}MIN_SAMPLE$[i].txt max_depth${MAX_DEPTH[j]}_min_sample_split${MIN_SAMPLE_SPLIT[k]}
else
	mkdir max_depth${MAX_DEPTH[j]}_min_sample_split${MIN_SAMPLE_SPLIT[k]}
	mv Resultado_${MAX_DEPTH[j]}DEPTH${MIN_SAMPLE_SPLIT[k]}MIN_SAMPLE$[i].txt max_depth${MAX_DEPTH[j]}_min_sample_split${MIN_SAMPLE_SPLIT[k]}
fi
done
done
done


