#!/bin/bash
n=0

for k in lung_{12..30}-label.nii.gz 
do
	((++n))
	./MetricTest $k pred_$n"_Segm.nii.gz"
done

