#!/bin/bash

ALGORITMO="florestaClass.py"

MAX_DEPTH=("" "Default" "9" "36")
N_ESTIMATORS=("" "Default" "50" "150")
N_JOBS=("" "Default" "-1" "-7")

for i in $(seq 5)
do
for j in $(seq 3)
do
for l in $(seq 3)
do
for m in $(seq 3)
do
	EXECUCAOPERF="sudo time perf stat -x | -e /power/energy-pkg/,/power/energy-cores/,/power/energy-gpu/"
	COMANDO2="sudo perf stat -x | -e instructions,cycles,cpu-clock,cpu-migrations,branches,branch-misses,context-switches,bus-cycles,cache-references,cache-misses,mem-loads,mem-stores,L1-dcache-stores"
	APP="python3.7 $ALGORITMO ${MAX_DEPTH[j]} ${N_ESTIMATORS[l]} ${N_JOBS[m]}"
        $EXECUCAOPERF $COMANDO2 $APP &>Resultado$[i]_DEPTH${MAX_DEPTH[j]}_N_ESTIMATORS${N_ESTIMATORS[l]}_N_JOBS${N_JOBS[m]}.txt
if [ -d "max_depth${MAX_DEPTH[j]}_n_estimators${N_ESTIMATORS[l]}_n_jobs${N_JOBS[m]}" ]
then
	mv Resultado$[i]_DEPTH${MAX_DEPTH[j]}_N_ESTIMATORS${N_ESTIMATORS[l]}_N_JOBS${N_JOBS[m]}.txt max_depth${MAX_DEPTH[j]}_n_estimators${N_ESTIMATORS[l]}_n_jobs${N_JOBS[m]}
else
	mkdir max_depth${MAX_DEPTH[j]}_n_estimators${N_ESTIMATORS[l]}_n_jobs${N_JOBS[m]}
	mv Resultado$[i]_DEPTH${MAX_DEPTH[j]}_N_ESTIMATORS${N_ESTIMATORS[l]}_N_JOBS${N_JOBS[m]}.txt max_depth${MAX_DEPTH[j]}_n_estimators${N_ESTIMATORS[l]}_n_jobs${N_JOBS[m]}
fi
done
done
done
done

