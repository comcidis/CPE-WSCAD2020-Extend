#!/bin/bash

EXEMPLO=("" "25")
ATRIBUTO=("" "5")

for i in $(seq 1)
do
for j in $(seq 1)
do
for k in $(seq 1)
do
	EXECUCAOPERF="sudo time perf stat -x | -e /power/energy-pkg/,/power/energy-cores/,/power/energy-gpu/"
	COMANDO2="sudo perf stat -x | -e instructions,cycles,cpu-clock,cpu-migrations,branches,branch-misses,context-switches,bus-cycles,cache-references,cache-misses,mem-loads,mem-stores,L1-dcache-stores,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-store-misses,LLC-stores,LLC-store-misses,LLC-loads,LLC-load-misses,minor-faults,page-faults"
	APP="python3.7 floresta_clasEntropia.py ${EXEMPLO[j]} ${ATRIBUTO[k]}"
        $EXECUCAOPERF $COMANDO2 $APP &>Resultado_${EXEMPLO[j]}${ATRIBUTO[k]}ATT$[i].txt
if [ -d "${EXEMPLO[j]}${ATRIBUTO[k]}entropia" ]
then
	mv Resultado_${EXEMPLO[j]}${ATRIBUTO[k]}ATT$[i].txt ${EXEMPLO[j]}${ATRIBUTO[k]}entropia
else
	mkdir ${EXEMPLO[j]}${ATRIBUTO[k]}entropia
	mv Resultado_${EXEMPLO[j]}${ATRIBUTO[k]}ATT$[i].txt ${EXEMPLO[j]}${ATRIBUTO[k]}entropia
fi
done
done
done

