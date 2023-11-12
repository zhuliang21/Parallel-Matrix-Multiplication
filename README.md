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


## TODO

- [ ] speedup curve
- [ ] slide for presentation
    - [ ] animation
- [ ] report


## General Idea: Multi Master-slave Tree
1. distribute data to each core on mutiple master-slave tree to the lowest level (recursive)
2. compute on the lowest level
3. collect data from the lowest level to the ROOT (recursive)

## Step 0: build the master-slave tree

Multi Master-slave Tree: (static)
```
0 -> [0, 1, 2, 3, 4, 5, 6]
---------------------------------
1 -> [7, 8, 9, 10, 11, 12, 13]
2 -> [14, 15, 16, 17, 18, 19, 20]
3 -> [21, 22, 23, 24, 25, 26, 27]
4 -> [28, 29, 30, 31, 32, 33, 34]
5 -> [35, 36, 37, 38, 39, 40, 41]
6 -> [42, 43, 44, 45, 46, 47, 48]
---------------------------------
7 -> [49, 50, 51, 52, 53, 54, 55]
......
```

- [x] `is_root` to check if current rank is root under current level
- [x] `is_worker` to check if current rank is worker under current level
- [x] `get_root_rank` to get root rank
- [x] `get_worker_ranks` to get worker rank

## Step 1: distribute data

Since after the addition or subtraction, the
Strassen submatrices multiplies have similar patterns: (submatrices of A) $\times$ (submatrices of B), we can first compute the submatrices of A and B on root, then send them to 7 cores, and then compute the 7 matrix multiplications on 7 cores.

|   $Mi_A$  |   $Mi_B$   | $Mi$ |
|:----:|:-----:| :-----:|
| $M1_A = (A_{11} + A_{22})$| $M1_B = (B_{11} + B_{22})$ | $M1 = M1_A \times M1_B$ |
| $M2_A = (A_{21} + A_{22})$| $M2_B = B_{11}$ | $M2 = M2_A \times M2_B$ |
| $M3_A = A_{11}$| $M3_B = (B_{12} - B_{22})$ | $M3 = M3_A \times M3_B$ |
| $M4_A = A_{22}$| $M4_B = (B_{21} - B_{11})$ | $M4 = M4_A \times M4_B$ |
| $M5_A = (A_{11} + A_{12})$| $M5_B = B_{22}$ | $M5 = M5_A \times M5_B$ |
| $M6_A = (A_{21} - A_{11})$| $M6_B = (B_{11} + B_{12})$ | $M6 = M6_A \times M6_B$ |
| $M7_A = (A_{12} - A_{22})$| $M7_B = (B_{21} + B_{22})$ |  $M7 = M7_A \times M7_B$ |

In this way, all $Mi_A$ and $Mi_B$ are same size, and can be sent to 7 cores for computation.


```
distribute_data(level, max_level){
    for level = 1, level < max_level, level ++ {
        if rank is leader unter current level {
            compute Mi_A and Mi_B
            send Mi_A and Mi_B to its 7 workers
        }
        if rank is worker unter current level {
            receive Mi_A and Mi_B from its leader
        }
    }
}
```


- [x] `prepare_strassen` to prepare `A` `B` into matrix array  `M_A[]` and `M_B[]`
- [x] distribute `M_A[]` and `M_B[]` submatrices to its workers
    - [x] `root_send_data` to send data to workers
    - [x] `worker_receive_data` to receive data from root

- [x] add input arguments `currentLevel` and `maxLevel`
- [x] add quit condition `if (currentLevel == maxLevel)`


- [ ] functions for root and worker relationship
    - [x] `get_leader_rank` to get root rank
    - [x] `get_worker_ranks` to get worker rank
    - [x] `is_leader` to check if current rank is root under current level
    - [x] `is_worker` to check if current rank is worker under current level

- [x] `distribute_data` to assign data to each core on mutiple master-slave tree to the lowest level (for-loop)
    - [x] leader devides and sends to its 7 workers
    - [x] worker receives data from its leader (2 matrix)
    - [x] for-loop to quit after reaching the lowest level (`max_level`), levle ++

## Step 2: compute on the lowest level

- [x] `matrix_multiply` to compute on the lowest level


## Step 3: collect results and combine into one matrix

```
collect_results(level, max_level){
    for level = max_level, level > 0, level -- {
        if rank is leader unter current level {
            receive data from its 7 workers
            combine them into one matrix
        }
        if rank is worker unter current level {
            send data to its leader
        }
    }
}
```
- [x] `collect_results` to collect data from the lowest level to the ROOT (for-loop)
    - [x] worker sends data (1 matrix) to leader under current level
    - [x] leader receives data from workers under current level and combine into one matrix
        - compute C11, C12, C21, C22
        - combine them into C
    - [x] for-loop to quit after reaching the ROOT (`level = 1`), level --

### Isues
- [x] when N is large, the ROOT send and receive will dead lock
    - dead lock: ROOT send data to itself, and wait for itself to receive
    - [x] `distributed_data`: when leader is ROOT, only send other 6 matrix to workers, and copy the first matrix; when worker is ROOT, take the copyed matrix to itself.
    - [x] `collect_results`: when worker is ROOT, only copy its result; when leader is ROOT, receive other 6 matrix from workers, and take the copyed matrix to itself.



Description (level1):
- ROOT devides and sends to its 7 workers (size: N/2)
- 7 workers compute and send back to 0 (size: N/2)
- 0 combine and return (size: N)

Description (level2):
- ROOT devides and sends to its 7 workers (size: N/2)
- 7 workers devides and sends to their 7x7 workers (size: N/4)
- 7x7 workers compute and send back to their 7 workers (size: N/4)
- 7 workers combine and send back to 0 (size: N/2)
- ROOT combine return (size: N)

Description (level3):
- ROOT devides and sends to its 7 workers (size: N/2)
- 7 workers devides and sends to their 7x7 workers (size: N/4)
- 7x7 workers devides and sends to their 7x7x7 workers (size: N/8)
- 7x7x7 workers compute and send back to their 7x7 workers (size: N/8)
- 7x7 workers combine and send back to their 7 workers (size: N/4)
- 7 workers combine and send back to 0 (size: N/2)
- ROOT combine and retrun (size: N)


```
distribute_data(level, max_level)
compute on the lowest level
collect_data(level, max_level)
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



## Appendix program
The program can be executed on the cluster by running the script `project3_appendix.sh`.
```
module load slurm
cd project3
sbatch project3_appendix.sh
```
The results will be stored in the file `output_appendix.txt`.