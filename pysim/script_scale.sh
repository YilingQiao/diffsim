#!/bin/bash
rm scale.txt	
mprof clean
for i in {1..20..1}
do
	mprof run python run_scale.py $i 
	mprof list
done