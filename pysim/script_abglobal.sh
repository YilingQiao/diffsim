#!/bin/bash
for i in {500..1000..500}
do
	for j in {1..1}
	do
		python run_absparse.py $i >> ./log/200205gours$i.txt
	done
done