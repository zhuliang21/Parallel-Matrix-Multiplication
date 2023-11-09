mpicc -o parallel_recursive_sum parallel_recursive_sum.c  
mpirun -np 8 ./parallel_recursive_sum

# mpicc -o recursiveSum recursiveSum.c  
# mpirun -np 8 ./recursiveSum