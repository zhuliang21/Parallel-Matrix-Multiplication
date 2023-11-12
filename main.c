#include "header.h"

void measure_time(int max_level, int rank, int size) {
    // set N values: 2^8, 2^10, 2^12
    int N_values[] = {1 << 8, 1 << 10, 1 << 12};
    for (int i = 0; i < sizeof(N_values) / sizeof(N_values[0]); i++) {
        int N = N_values[i];

        // generate matrix A and B on root process
        Matrix *A = NULL, *B = NULL;
        if (rank == ROOT) {
            // generate matrix A and B on 
            A = generate_matrix(N, N);  
            B = generate_matrix(N, N);
        } 
        
        // test strassen_parallel and record the time
        double start_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        Matrix* result_by_strassen_parallel = strassen_multiply_parallel(A, B, N, max_level);
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        if (rank == ROOT) {
            int print_N = get_n_from_size(N);
            printf("N = 2^%d, max_level = %d, cores = %d, time = %f\n", print_N, max_level, size, end_time - start_time);
        }
        if (rank == ROOT) {
            // // result by navie method
            // Matrix* result_by_navie = strassen_multiply_serial(A, B);
            // // confirm the results are the same
            // int is_same = quick_check_result(result_by_strassen_parallel, result_by_navie, 0.0001);
            // if (is_same) {
            //     printf("Pass results check!\n");
            // } else {
            //     printf("The results are different!\n");
            // }
            // free memory
            free_matrix(A);
            free_matrix(B);
            // free_matrix(result_by_navie);
            free_matrix(result_by_strassen_parallel);
        }
    }
}

// Comfirm that the ring algorithm works by comparing the result with the naive algorithm
void result_check(int size, int rank) {
    int N = 4;
    Matrix *A, *B, *result_by_naive, *result_by_strassen;

    // 1. generate matrix A and B (only in root process)
    A = NULL; B = NULL;
    if(rank == ROOT) {
        printf("\n");
        printf("Test on small matrix to check the correctness of the algorithm\n");
        printf("N = %d, P = %d : \n", N, size);

        A = generate_matrix(N, N);
        B = generate_matrix(N, N);
        printf("Matrix A: \n");
        print_matrix(A);
        printf("Matrix B: \n");
        print_matrix(B);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // 2. call naive_matrix_multiply()
    if(rank == ROOT) {
        result_by_naive = matrix_multiply(A, B);
        printf("Result by naive:\n");
        print_matrix(result_by_naive);
    }

    // 3. call ring_matrix_multiply()
    MPI_Barrier(MPI_COMM_WORLD);
    result_by_strassen = strassen_multiply_parallel(A, B, N, 2);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == ROOT) {
        printf("Result by strassen parallel (level-2):\n");
        print_matrix(result_by_strassen);
    }

    // 4. check if the results are the same
    if (rank == ROOT) {
        double tolerance = 1e-6;
        if (quick_check_result(result_by_naive, result_by_strassen, tolerance)) {
            printf("Pass results check! Two algorithms give the same result.\n");
        } else {
            printf("Results differ!\n");
        }
        free_matrix(result_by_naive);
        free_matrix(result_by_strassen);    
    }
    free_matrix(A);
    free_matrix(B);
}


int main(int argc, char** argv) {
    // initialize MPI environment
    MPI_Init(&argc, &argv);

    // get the number of processes and the rank of this process
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int max_level;

    if (argc == 2) {
        if (strcmp(argv[1], "level_1") == 0) {
            max_level = 1;
            measure_time(max_level, rank, size);
        }
        else if (strcmp(argv[1], "level_2") == 0) {
            max_level = 2;
            measure_time(max_level, rank, size);
        }
        else if (strcmp(argv[1], "level_3") == 0) {
            max_level = 3;
            measure_time(max_level, rank, size);
        }
        else if (strcmp(argv[1], "result_check") == 0) {
            result_check(size, rank);
        } else {
            printf("Please enter  [level_1], [level_2], [level_3] or [result_check] as the argument \n");
        }
    } else {
        max_level = 1;
        measure_time(max_level, rank, size);
    }
    // 终止MPI环境
    MPI_Finalize();
    return 0;
}
