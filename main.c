#include "header.h"

int main(int argc, char** argv) {
    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    // print size of MPI_COMM_WORLD
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 获取当前进程的rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 1 << 5;
    int max_level = 1;

    // 创建矩阵A和B，只在根进程进行
    Matrix *A = NULL, *B = NULL;
    if (rank == ROOT) {
        // generate matrix A and B on 
        A = generate_matrix(N, N);  
        B = generate_matrix(N, N);
    } 

    // test strassen_parallel
    Matrix* result_by_strassen_parallel = strassen_parallel(A, B, N, max_level);

    if (rank == ROOT) {
        // result by navie method
        Matrix* result_by_navie = strassen_multiply_serial(A, B);
        // confirm the results are the same
        int is_same = quick_check_result(result_by_strassen_parallel, result_by_navie, 0.0001);
        if (is_same) {
            printf("Pass results check!\n");
        } else {
            printf("The results are different!\n");
        }
        // free memory
        free_matrix(A);
        free_matrix(B);
        free_matrix(result_by_navie);
        free_matrix(result_by_strassen_parallel);
    }

    // 终止MPI环境
    MPI_Finalize();
    return 0;
}
