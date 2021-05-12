#!/bin/bash

EXEMPLO=("" "2" "3" "5")
ATRIBUTO=("" "5" "10" "21")

for i in $(seq 2)
do
for j in $(seq 3)
do
for k in $(seq 3)
do
	EXECUCAOPERF="sudo time perf stat -x | -e /power/energy-pkg/,/power/energy-cores/,/power/energy-gpu/"
	COMANDO2="sudo perf stat -x | -e instructions,cycles,cpu-clock,cpu-migrations,branches,branch-misses,context-switches,bus-cycles,cache-references,cache-misses,mem-loads,mem-stores,L1-dcache-stores,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-store-misses,LLC-stores,LLC-store-misses,LLC-loads,LLC-load-misses,minor-faults,page-faults"
	APP="python3.7 arvore4.py ${EXEMPLO[j]} ${ATRIBUTO[k]}"
        $EXECUCAOPERF $COMANDO2 $APP &>Resultado_${EXEMPLO[j]}${ATRIBUTO[k]}ATT$[i].txt
if [ -d "${EXEMPLO[j]}${ATRIBUTO[k]}" ]
then
	mv Resultado_${EXEMPLO[j]}${ATRIBUTO[k]}ATT$[i].txt ${EXEMPLO[j]}${ATRIBUTO[k]}
else
	mkdir ${EXEMPLO[j]}${ATRIBUTO[k]}
	mv Resultado_${EXEMPLO[j]}${ATRIBUTO[k]}ATT$[i].txt ${EXEMPLO[j]}${ATRIBUTO[k]}
fi
done
done
done
