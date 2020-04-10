#!/bin/bash
echo "t" > data1.csv
echo "t" > data2.csv
for (( N=0; N<1000; N++ ))
do
	OUTPUT1=$(./prog 1024 $RANDOM)
	echo "$OUTPUT1" >> data1.csv
	if [ $(($N%100)) -eq 0 ]
	then
		echo "$N"
	fi
done

for (( N=0; N<1000; N++ ))
do
	OUTPUT2=$(./smprog 1024 $RANDOM)
	echo "$OUTPUT2" >> data2.csv
	if [ $(($N%100)) -eq 0 ]
	then
		echo "$N"
	fi
done

