#!/bin/bash
mkdir qr_out

for i in {100..300..100}
do
	for j in {1..5}
	do
		python run_abqr.py $i >> ./log/compabqr$i.txt
	done
done