#!/bin/bash
mkdir sparse_out

for i in {100..300..100}
do
	for j in {1..5}
	do
		python run_absparse.py $i >> ./log/compabsparse$i.txt
	done
done