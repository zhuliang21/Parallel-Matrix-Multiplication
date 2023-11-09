#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ROOT 0
#define NUM_WORKERS 2

// function declaration
int* generate_data(int data_size);
void print_data(int* data, int data_size);
int calculate_sum(int* data, int data_size);
int is_root(int rank, int currentLevel);
int is_worker(int rank, int currentLevel);
int* get_worker_ranks(int rank);
int get_root_rank(int rank);
int recursive_sum_serial(int* data, int data_size, int level, int max_level);
void root_send_data(int* data, int data_size, int rank);



// function to generate data
int* generate_data(int data_size) {
    int* data = (int*)malloc(sizeof(int) * data_size);
    for (int i = 0; i < data_size; i++) {
        data[i] = i;
    }
    return data;
}

// function to print data
void print_data(int* data, int data_size) {
    for (int i = 0; i < data_size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

// function to calculate sum of data
int calculate_sum(int* data, int data_size) {
    int sum = 0;
    for (int i = 0; i < data_size; i++) {
        sum += data[i];
    }
    return sum;
}

// function to calculate sum of data recursively on single core: recursive_sum_serial
int recursive_sum_serial(int* data, int data_size, int level, int max_level) {
    if (level == max_level) {
        return calculate_sum(data, data_size);
    } else {
        int mid = data_size / 2;
        int sum1 = recursive_sum_serial(data, mid, level + 1, max_level);
        int sum2 = recursive_sum_serial(data + mid, data_size - mid, level + 1, max_level);
        // join sum1 and sum2 into sum array
        int* sum = (int*)malloc(sizeof(int) * 2);
        sum[0] = sum1;
        sum[1] = sum2;
        return calculate_sum(sum, 2);
    }
}

// function to check if the process is master: is_root
int is_root(int rank, int currentLevel) {
    return rank < currentLevel;
}

// function to check if the process is worker: is_worker
int is_worker(int rank, int currentLevel) {
    return rank < pow(NUM_WORKERS, currentLevel);
}

// function to get workers' ranks of each root, (2 workers): get_worker_ranks
int* get_worker_ranks(int rank) {
    int* workerRanks = (int*)malloc(sizeof(int) * NUM_WORKERS);
    for (int i = 0; i < NUM_WORKERS; i++) {
        workerRanks[i] = rank * NUM_WORKERS + i;
    }
    return workerRanks;
}

// function to get root rank of current process: get_root_rank
int get_root_rank(int rank) {
    // root the rank/NUM_WORKERS and get the integer part
    return rank / NUM_WORKERS;
}


// function to send data to its workers: root_send_result
void root_send_data(int* data, int data_size, int rank) {
    int* workerRanks = get_worker_ranks(rank);
    for (int i = 0; i < NUM_WORKERS; i++) {
        // send data to its workers by Isend and wait for all Isend to complete
        MPI_Isend(data, data_size, MPI_INT, workerRanks[i], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}
   




// function to send data to its root: worker_send_data

// function to receive data from its workers: root_receive_result





// main function to test the program
int main(int argc, char** argv) {
    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    // print size of MPI_COMM_WORLD
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 获取当前进程的rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int level = 1;
    int data_size = 8;
    int* data;
    if (rank == 0) {
        printf("rank = %d\n", rank);
        // generate data of size 8, calculate sum of data by recursive_sum_serial and print the result
        data_size = 8;
        data = generate_data(data_size);
        print_data(data, data_size);
        free(data);
    } 
    if (is_root(rank, level)) {
        printf("root rank = %d\n", rank);
        // print its workers' ranks
        int* workerRanks = get_worker_ranks(rank);
        printf("worker ranks: ");
        for (int i = 0; i < NUM_WORKERS; i++) {
            printf("%d ", workerRanks[i]);
        }
        printf("\n");
        
        // send data to its workers
        root_send_data(data, data_size, rank);
    }
    if (is_worker(rank, level)) {
        printf("worker rank = %d\n", rank);
        // print its root rank
        printf("its root rank: %d\n", get_root_rank(rank));
    }

    // 结束MPI环境
    MPI_Finalize();
    return 0;
}


