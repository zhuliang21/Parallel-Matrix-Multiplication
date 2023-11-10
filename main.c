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
    int max_level = 1;
    int level;

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

    // Step 1: divide matrix to workers, until level = max_level
    for (level = 1; level <= max_level; level++) {
        distribute_data(&M_A, &M_B, A, B, local_n, level);
        //  updata A B with M_A M_B
        A = M_A;
        B = M_B;
        if (level < max_level) {
            local_n = local_n / 2; // don't update local_n on the last level
        }
        printf("level %d\n", level);
    }
    level = max_level; // set level to max_level

    // Step 2: compute C = A * B
    Matrix* C = matrix_multiply(A, B);
    printf("matrix C: on process %d\n", rank);
    print_matrix(C);
    free_matrix(A);
    free_matrix(B);

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 3: collect results from workers and combine into C, until level = 1
    Matrix* M_C;
    collect_results(&M_C, C, local_n, level);


    // 终止MPI环境
    MPI_Finalize();
    return 0;
}
