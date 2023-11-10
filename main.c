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

    int N = 1 << 2;
    int local_n = N / 2;


    // 创建矩阵A和B，只在根进程进行
    Matrix *A = NULL, *B = NULL;
    if (rank == ROOT) {
        // generate matrix A and B on 
        A = generate_matrix(N, N);  
        B = generate_matrix(N, N);
        printf("generate matrix A and B on leader process\n");
        printf("matrix A:\n");
        print_matrix(A);
        printf("matrix B:\n");
        print_matrix(B);
    }
    Matrix *M_A = NULL, *M_B = NULL;
    distribute_data(&M_A, &M_B, A, B, local_n, 1);
    // compute C on each process
    MPI_Barrier(MPI_COMM_WORLD);
    printf("process %d: compute C\n", rank);
    // 终止MPI环境
    MPI_Finalize();
    return 0;
}
