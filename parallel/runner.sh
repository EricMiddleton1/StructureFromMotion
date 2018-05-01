#!/bin/bash

num_proc=$1

echo "Starting script with max processing elements $(bc <<< "2^$num_proc")"

echo "num_proc, time_load, time_load_xfer, time_feature, time_feature_xfer, time_covis, time_covis_xfer, time_motion" | tee results.csv

for i in `seq 0 $num_proc`; do
	p=$(bc <<< "2^$i")
	n=$(bc <<< "($p-1) / 16 + 1")
	t=$(bc <<< "($p-1) % 16 + 1")
	#echo "P=$p, N=$n, T=$t"
	mpirun -n $n -bind-to none ./sfm $t | tee -a results.csv
done
