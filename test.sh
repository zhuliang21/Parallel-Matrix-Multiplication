#!/bin/bash
make clean
make

mpirun -np 7 ./test 