#!/bin/bash
for i in {50..150..50}
do
	for j in {1..2}
	do
		python run_abqr.py $i >> ./log/compabqr$i.txt
	done
done