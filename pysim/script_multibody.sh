#!/bin/bash
rm multibody.txt
mprof clean
for i in {1..20..1}
do
	mprof run python run_multibody.py $i 
	mprof list
done