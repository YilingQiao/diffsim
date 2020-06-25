#!/bin/bash
rm scale.txt	

# to profile memory
# mprof clean
# for i in {1..20..1}
# do
#	mprof run python run_scale.py $i 
#	mprof list
# done

for i in {1..20..1}
do
	python run_scale.py $i 
done