#!/bin/bash
n=1
for k in images/*.mhd
do
	name=${k#images/}
	echo "Converting image $n"
	c3d $k -o images/lung_$n.nii.gz
	c3d labels/$name -o labels/lung_$n-label.nii.gz
	((n++))
done
