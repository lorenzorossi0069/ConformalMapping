#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "possible scripts in this folder:"
	ls -l *.py
	exit 1
fi

source ./bin/activate

python3 ./$1




