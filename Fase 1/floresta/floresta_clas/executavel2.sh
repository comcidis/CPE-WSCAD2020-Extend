#!/bin/bash

EXEMPLO=("" "100")
ATRIBUTO=("" "21")


for i in $(seq 2)
do
for j in $(seq 1)
do
for k in $(seq 1)
do
	EXECUCAOPERF="sudo time perf stat -x | -e /power/energy-pkg/,/power/energy-cores/,/power/energy-gpu/"
	COMANDO2="sudo perf stat -x | -e instructions,cycles,cpu-clock,cpu-migrations,branches,branch-misses,context-switches,bus-cycles,cache-references,cache-misses,mem-loads,mem-stores,L1-dcache-stores,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-store-misses,LLC-stores,LLC-store-misses,LLC-loads,LLC-load-misses,minor-faults,page-faults"
	APP="python3.7 floresta_class_2.py ${EXEMPLO[j]} ${ATRIBUTO[k]}"
        $EXECUCAOPERF $COMANDO2 $APP &>Resultado_${EXEMPLO[j]}${ATRIBUTO[k]}ATT$[i].txt
if [ -d "${EXEMPLO[j]}${ATRIBUTO[k]}categorico" ]
then
	mv Resultado_${EXEMPLO[j]}${ATRIBUTO[k]}ATT$[i].txt ${EXEMPLO[j]}${ATRIBUTO[k]}categorico
else
	mkdir ${EXEMPLO[j]}${ATRIBUTO[k]}categorico
	mv Resultado_${EXEMPLO[j]}${ATRIBUTO[k]}ATT$[i].txt ${EXEMPLO[j]}${ATRIBUTO[k]}categorico
fi
done
done
done
