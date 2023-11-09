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
    int level = 1; 

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

    if (is_leader(rank, level)) {
        printf("process %d is leader under level %d\n", rank, level);
        //print its worker rank
        int* worker_rank = get_worker_rank(rank);
        printf("process %d's worker rank:\n", rank);
        print_data(worker_rank, NUM_TASKS);
        // prepare strassen
        Matrix* M_A[NUM_TASKS];
        Matrix* M_B[NUM_TASKS];
        prepare_strassen(A, B, M_A, M_B);
        // send M_A and M_B to its workers by MPI_Isend
        for (int i = 0; i < NUM_TASKS; i++) {
            MPI_Request requests[NUM_TASKS * 2];
            int request_count = 0;        
            MPI_Isend(M_A[i]->data, M_A[i]->rows * M_A[i]->cols, MPI_DOUBLE, worker_rank[i], 0, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Isend(M_B[i]->data, M_B[i]->rows * M_B[i]->cols, MPI_DOUBLE, worker_rank[i], 0, MPI_COMM_WORLD, &requests[request_count++]);
        }
        printf("send M_A and M_B to its workers by MPI_Isend\n");
    }

    if (is_worker(rank, level)) {
        printf("process %d is worker under level %d\n", rank, level);
        //print its leader rank
        int leader_rank = get_leader_rank(rank);
        printf("process %d's leader rank: %d\n", rank, leader_rank);
        double *local_A = (double*) malloc(local_n * local_n * sizeof(double));
        double *local_B = (double*) malloc(local_n * local_n * sizeof(double));

        // receive M_A and M_B from its leader by MPI_Irecv
        MPI_Request recv_requests[2];
        int recv_count = 0;

        MPI_Irecv(local_A, local_n * local_n, MPI_DOUBLE, leader_rank, 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
        MPI_Irecv(local_B, local_n * local_n, MPI_DOUBLE, leader_rank, 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
        MPI_Waitall(recv_count, recv_requests, MPI_STATUS_IGNORE);

        Matrix* M_A = NULL;
        Matrix* M_B = NULL;
        // 分配和构建矩阵结构
        M_A = malloc(sizeof(Matrix));
        M_A->data = local_A;  
        M_A->rows = local_n; 
        M_A->cols = local_n;

        M_B = malloc(sizeof(Matrix));
        M_B->data = local_B; 
        M_B->rows = local_n; 
        M_B->cols = local_n;       

        printf("receive M_A and M_B from its leader by MPI_Irecv\n");
        printf("M_A:\n");
        print_matrix(M_A);
        printf("M_B:\n");
        print_matrix(M_B);
    }
    
    // 终止MPI环境
    MPI_Finalize();
    return 0;
}
