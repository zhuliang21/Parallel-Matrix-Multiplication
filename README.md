# Project 4.1 - Parallel Matrix Multiplication using Strassen's Algorithm

## Project Description
Design an algorithm to multiply two large matrices $A, B \in \mathbb{R}^{N\times N}$ using  using the Strassen's algorithm on a parallel system available to you. You can assume the matrix size $N = 2^\alpha$ where $\alpha$ is an integer that should be bigger than 10 (but small enough to fit in your system). For example, if you are given a matrix of size $2^{14} \times 2^{14}$, in Level-1 Strassen, you will compute $7^1$ matrix multiplications involving matrices of size $2^{13} \times 2^{13}$. To do Level-2 Strassen, you will have $7^2 = 49$ matrix multiplications. To do Level-3 Strassen, you will have $7^3 = 343$ matrix multiplications. Your computer system is asssumed to have multiple of $7$ processor... thus, you will have $7p$ processors.

In this project, you carry three (3) levels of Strassen's algorithm, and you may select any ramdom numbers (not all are zeros, of course) as your matrix elements.

Please complete the following:

1. Design the algorithms on how to distribute the sub-matrices to your participating cores with the objectives of minimizing memory use and communication costs.
2. Test the performance using on $7^{\beta}$ with $\beta = 1, 2, 3, $ for matrix of size $N = 2^\alpha$ with $\alpha = 8, 10, 12$ 
3. Collect the performance results and analyze them.
4. Plot the speedup curves.
5. Comment on your performance results.

 $P = 7^{Level}$

 $N = 256, 1024, 4096$

 $Level = 1, 2, 3$

 Total situations:  $3 \times 3 = 9$

## Questions
- How many situations do we need to test for a given $N$?
```
| ------- | Level-1  | Level-2 | Level-3  |
| ------- | 7 tasks  | 49 tasks| 343 tasks|
| P = 7   |  even    | even    |  even    |
| P = 14  |    ?     | not even|  not even|
| P = 21  |    ?     | not even|  not even|
| P = 28  |    ?     | not even|  not even|
```

- Under Level-1, only 7 cores are needed, then how to test P = 14, 21, 28?
    - the speedup measure for these cases is strange. Not all cores are used.
- Why we need the core number to be multiple of 7 anyway?
    - for P = 14, 21, 28, the tasks cannot be evenly distributed to cores.
- When tasks number not equal to core number, the assigenment will be complex:
    - will need to determine how many tasks and task id for each core, before sending tasks.
    - sending and receiving will be complex, mutiple rounds of sending and receiving.
    - speedup will be lower than (tasks number = core number).
- Should I make the parallel algorithm recursive?
    - Otherwise, I can write 3 subroutines for Level-1, Level-2, Level-3.
- Should I use multiple master-slave trees when level > 1?
    - Otherwise, I can use one master-slave tree.
- My current idea:
    - root generates A B, and split them into 7, 49, 343 tasks. 
        - only one root core, 
        - only root has full A B
        - only decompose A B on root (not decompose recursively)
    - root sends tasks and related sub-matrices to each core. (7, 14, 21, 28 cores)
    - each core computes the tasks assigned to it.
    - each core sends the results back to root.
    - root combines the results into C.
- How about 2-level master-slave network?
## TODO
### Strassen's algorithm (with option to Level-1, Level-2, Level-3)
#### implement Straaaen on a single core `strassen_multiply_serial()` 

P = 1, recursive, variable level.
- [x] `split_matrix()`
- [x] `combine_matrix()`
- [x] `matrix_add()`
- [x] `matrix_subtract()`
- [x] `matrix_multiply()`
- [x] `strassen_multiply_serial()`
- [x] `strassen_multiply_serial_recursive()`  
- [x] `main()`
    - generate matrices A B
    - call `matrix_multiply()`
    - call `strassen_multiply_serial()` or `strassen_multiply_serial_recursive()`
    - check if results are the same

Pesudo code for `strassen_multiply_serial_recursive()`:
```
f(A, B, level, max_level){
    if (level = max_level) {
        return simple_multiply(A, B)
    }
    else {
        split A B into submatrices
        assemble M1_A to M7_A, M1_B to M7_B
        M1 = f(M1_A, M1_B, level+1, max_level)
        M2 = f(M2_A, M2_B, level+1, max_level)
        M3 = f(M3_A, M3_B, level+1, max_level)
        M4 = f(M4_A, M4_B, level+1, max_level)
        M5 = f(M5_A, M5_B, level+1, max_level)
        M6 = f(M6_A, M6_B, level+1, max_level)
        M7 = f(M7_A, M7_B, level+1, max_level)
        combine M1 to M7 into C
        return C
    }
}
```

####  test if Seawulf allow 343 cores call, YES


####  implement Strassen on multiple cores with level-1 on 7 cores 
`strassen_multiply_parallel_level_1()`

P = 7, level = 1.
- [x] root core splits A B into submatrices.(`split_matrix`)
- [x] root core to send submatrices size to 7 cores. (`MPI_Bcast`)
- [x] root core sends submatricesm combination to 7 cores based on rank (include root). (`MPI_ISend`, `MPI_IRecv`)
- [x] 7 cores compute 7 matrix multiplications. (`matrix_multiply`)
- [x] 7 cores send results back to root core. (`MPI_ISend`, `MPI_IRecv`)
- [x] root core combines results into C. (`combine_matrix`)



Since after the addition or subtraction, the
Strassen submatrices multiplies have similar patterns: (submatrices of A) $\times$ (submatrices of B), we can first compute the submatrices of A and B on root, then send them to 7 cores, and then compute the 7 matrix multiplications on 7 cores.

|   $Mi_A$  |   $Mi_B$   | 
|:----:|:-----:|
| $M1_A = (A_{11} + A_{22})$| $M1_B = (B_{11} + B_{22})$ |
| $M2_A = (A_{21} + A_{22})$| $M2_B = B_{11}$ |
| $M3_A = A_{11}$| $M3_B = (B_{12} - B_{22})$ |
| $M4_A = A_{22}$| $M4_B = (B_{21} - B_{11})$ |
| $M5_A = (A_{11} + A_{12})$| $M5_B = B_{22}$ |
| $M6_A = (A_{21} - A_{11})$| $M6_B = (B_{11} + B_{12})$ |
| $M7_A = (A_{12} - A_{22})$| $M7_B = (B_{21} + B_{22})$ |

In this way, all $Mi_A$ and $Mi_B$ are same size, and can be sent to 7 cores for computation.

- use tag to identify M1 to M7 results. (Level) (M1-7) (A-B)

#### implement Strassen on multiple cores with an option to level-1(7 cores), level-2(49 cores) (resursive) `strassen_multiply_parallel()`

- [x] add function `prepare_strassen` to prepare M submatrices in array for Strassen's algorithm
- [x] add function `worker_receive_submatrices` to recieve M_A and M_B submatrices from leader
- [x] add function `worker_send_results` to send M1 to M7 back to root
- [x] add function `root_send_submatrices` to send M_A and M_B submatrices to workers
- [x] add function `root_receive_results` to receive M1 to M7 from workers

Try 

- [ ] add input arguments `currentLevel` and `maxLevel`
- [ ] add quit condition `if (currentLevel == maxLevel)`

#### `parallel_recursive_sum` 
to testrun if the parallel recursive function works
- [ ] functions for root and worker relationship
    - [x] `get_root_rank` to get root rank
    - [x] `get_worker_ranks` to get worker rank
    - [x] `is root` to check if current rank is root under current level
    - [x] `is worker` to check if current rank is worker under current level

- [x] `distribute_data` to assign data to each core on mutiple master-slave tree to the lowest level
    - [ ] `root_send_data` to send data to workers
    - [ ] `worker_receive_data` to receive data from root
    


```
f(data, data_size, level, max_level){
    if (level = max_level) {
        return sum(data, data_size)
    }
    else {
        split data into subdata data1, data2
        half_size = data_size / 2
        sum1 = f(data1, half_size, level+1, max_level)
        sum2 = f(data2, half_size, level+1, max_level)
        return sum1 + sum2
    }
}
```


Multiple Master-slave Tree:
```
0 -> [0, 1, 2, 3, 4, 5, 6]
1 -> [7, 8, 9, 10, 11, 12, 13]
2 -> [14, 15, 16, 17, 18, 19, 20]
3 -> [21, 22, 23, 24, 25, 26, 27]
4 -> [28, 29, 30, 31, 32, 33, 34]
5 -> [35, 36, 37, 38, 39, 40, 41]
6 -> [42, 43, 44, 45, 46, 47, 48]
```
Description (level1):
- 0 devides and sends to its 7 workers (size: N/2)
- 7 workers compute and send back to 0 (size: N/2)
- 0 combine and return (size: N)

Description (level2):
- 0 devides and sends to its 7 workers (size: N/2)
- 7 workers devides and sends to their 7x7 workers (size: N/4)
- 7x7 workers compute and send back to their 7 workers (size: N/4)
- 7 workers combine and send back to 0 (size: N/2)
- 0 combine return (size: N)

Description (level3):
- 0 devides and sends to its 7 workers (size: N/2)
- 7 workers devides and sends to their 7x7 workers (size: N/4)
- 7x7 workers devides and sends to their 7x7x7 workers (size: N/8)
- 7x7x7 workers compute and send back to their 7x7 workers (size: N/8)
- 7x7 workers combine and send back to their 7 workers (size: N/4)
- 7 workers combine and send back to 0 (size: N/2)
- 0 combine and retrun (size: N)


```
// 主递归函数，用于实现多级master-slave模型
void recursiveMasterSlave(int currentLevel, int maxLevel, int rank, int numTasks) {
    if (currentLevel == maxLevel) {
        // 执行基本任务
        executeBaseTask(rank);
        return;
    }

    if (isMaster(rank, currentLevel)) {
        // 当前核心是master
        distributeTasks(currentLevel, rank, numTasks);
        collectResults(currentLevel, rank, numTasks);
    } else {
        // 当前核心是worker
        receiveTaskData(rank);
        recursiveMasterSlave(currentLevel + 1, maxLevel, rank, numTasks);
        sendResultData(rank);
    }
}
```
### Measruing the execution time




## Project Structure
```
project4/
│
├── header.h               # Header file 
├── main.c                 # Main program entry
├── functions.c            # Implementation of functions
├── make                   # Makefile for building the project
├── test.sh                # Script for testing on local machine
├── project4.sh            # Script to execute on the cluster
|── project4_appendix.sh   # Script to execute on the cluster for appendix
├── output.txt             # Output file
├── output_appendix.txt    # Output file for appendix
├── report.pdf             # Report PDF file
└── README.md              # This documentation file

```

## How to run the program
The program can be executed on the cluster by running the script `project3.sh`. 
The script will compile the program and execute it. 

I used the following commands to run the program on the node `short-96core` on the `milan` login node:
```
module load slurm
cd project3
chmod +x test.sh
sbatch project3.sh
```
The results will be stored in the file `output.txt`.

The output records three parts:
- test on small matrices and comfirm both algorithms are correct; 
- measure the execution time for Ring algorithm (P>1);
- measure the execution time for Naive algorithm (P=1)


## Appendix program
The program can be executed on the cluster by running the script `project3_appendix.sh`.
```
module load slurm
cd project3
sbatch project3_appendix.sh
```
The results will be stored in the file `output_appendix.txt`.