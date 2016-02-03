#!/bin/bash
mkdir -p tests
cd tests
for i in `seq 2 27`; do
	../build/src/DeployTest/DeployTest -b 3 -d 0,1 -e $i,0
	../build/src/DeployTest/DeployTest -b 3 -d 0,1 -e $i,1
	../build/src/DeployTest/DeployTest -b 3 -d 0,1 -e $i,2
done

