#!/bin/bash
#  run.sh
#  Created by Steve DengZishi on 03/04/18.
#  Copyright © 2018 Steve DengZishi. All rights reserved.
echo -e "\nRun.sh Producing MPI parallel comp data      Version 1.0"
echo -e "       Copyright © 2018 Steve DengZishi  New York University\n"

unknown=1
declare -a procNum=(1 2 10 20 40)
mpicc -g -Wall -std=c99 -o gs gs.c

for((i=1;i<5;i++))
do
	unknown=$(($unknown*10))
	#echo -e $unknown
	#echo ${procNum[$(($i-1))]}
	#generate files
	./gengs $unknown 0.001
	for((j=0;j<5;j++))
	do
		process=${procNum[$j]}
		if ((process<=$unknown))
		then
			echo "Number of unknown" $unknown "Number of processes" $process
			time mpirun -n $process ./gs $unknown.txt
		fi
	done
done
