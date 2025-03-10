#include "header.h"

// Function to get n value from size
int get_n_from_size(int N) {
    int n = 0;
    while (N >>= 1) {
        n++;
    }
    return n;
}

// Function to allocate memory for the matrix
Matrix* allocate_matrix(int rows, int cols) {
    Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
    if (matrix == NULL) {
        return NULL; // Memory allocation failed
    }
    
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (double*) malloc(rows * cols * sizeof(double));
    if (matrix->data == NULL) {
        free(matrix);
        return NULL; // Memory allocation failed
    }

    return matrix;
}

// Function to generate matrix based on row and conlumn
Matrix* generate_matrix(int rows, int cols) {
    Matrix* matrix = allocate_matrix(rows, cols);
    if (matrix == NULL) {
        return NULL; // Memory allocation failed
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix->data[i * cols + j] = 2.0 * ((double)rand() / (double)(RAND_MAX)) - 1.0; // Random number between (-1, 1]
        }
    }

    return matrix;
}

// Function to free the matrix memory
void free_matrix(Matrix* matrix) {
    if (matrix == NULL) {
        return;
    }

    free(matrix->data); // Free the continuous memory block
    free(matrix);      // Free the struct
}

// Function to print the matrix
void print_matrix(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%.2f ", matrix->data[i * matrix->cols + j]);
        }
        printf("\n");
    }
}

// Function to split the matrix into 4 submatrices
void split_matrix(const Matrix *original, Matrix *block11, Matrix *block12, Matrix *block21, Matrix *block22) {
    int new_rows = original->rows / 2;
    int new_cols = original->cols / 2;

    // 分配子矩阵
    *block11 = *allocate_matrix(new_rows, new_cols);
    *block12 = *allocate_matrix(new_rows, new_cols);
    *block21 = *allocate_matrix(new_rows, new_cols);
    *block22 = *allocate_matrix(new_rows, new_cols);

    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            // 填充子矩阵 block11
            block11->data[i * new_cols + j] = original->data[i * original->cols + j];
            // 填充子矩阵 block12
            block12->data[i * new_cols + j] = original->data[i * original->cols + (j + new_cols)];
            // 填充子矩阵 block21
            block21->data[i * new_cols + j] = original->data[(i + new_rows) * original->cols + j];
            // 填充子矩阵 block22
            block22->data[i * new_cols + j] = original->data[(i + new_rows) * original->cols + (j + new_cols)];
        }
    }
}

// Function to add two matrices
Matrix* matrix_add(const Matrix* A, const Matrix* B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        // Sizes don't match, cannot add
        return NULL;
    }

    Matrix* result = allocate_matrix(A->rows, A->cols);
    if (result == NULL) {
        // Memory allocation failed
        return NULL;
    }

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            int idx = i * A->cols + j;
            result->data[idx] = A->data[idx] + B->data[idx];
        }
    }

    return result;
}

// Function to subtract two matrices
Matrix* matrix_subtract(const Matrix* A, const Matrix* B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        // 矩阵维度不匹配
        return NULL;
    }
    
    Matrix* result = allocate_matrix(A->rows, A->cols);
    if (result == NULL) {
        return NULL; // 内存分配失败
    }
    
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            result->data[i * A->cols + j] = A->data[i * A->cols + j] - B->data[i * B->cols + j];
        }
    }
    
    return result;
}

// Function to multiply two matrices
Matrix* matrix_multiply(const Matrix* A, const Matrix* B) {
    if (A->cols != B->rows) {
        // A的列数必须等于B的行数才能进行乘法
        return NULL;
    }

    Matrix* result = allocate_matrix(A->rows, B->cols);
    if (result == NULL) {
        return NULL; // 内存分配失败
    }
    
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            result->data[i * result->cols + j] = 0;
            for (int k = 0; k < A->cols; k++) {
                result->data[i * result->cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
        }
    }
    
    return result;
}
// Function to copy a matrix
Matrix* copy_matrix(const Matrix* original) {
    Matrix* result = allocate_matrix(original->rows, original->cols);
    if (result == NULL) {
        return NULL; // 内存分配失败
    }

    memcpy(result->data, original->data, original->rows * original->cols * sizeof(double));
    return result;
}

// Function to combine 4 submatrices into a single matrix
Matrix* combine_matrix(Matrix *C11, Matrix *C12, Matrix *C21, Matrix *C22) {
    // 首先检查子矩阵是否具有相同的大小
    if (C11->rows != C12->rows || C11->cols != C21->cols || 
        C21->rows != C22->rows || C12->cols != C22->cols) {
        // 如果不一致，则释放已经分配的内存并返回NULL
        return NULL;
    }

    int halfSize = C11->rows;
    int fullSize = halfSize * 2;
    Matrix* result = allocate_matrix(fullSize, fullSize);
    if (!result) {
        // 如果内存分配失败，则释放已经分配的内存并返回NULL
        return NULL;
    }

    // 将子矩阵的值复制到结果矩阵的对应位置
    for (int i = 0; i < halfSize; ++i) {
        for (int j = 0; j < halfSize; ++j) {
            result->data[i * fullSize + j] = C11->data[i * halfSize + j];
            result->data[i * fullSize + j + halfSize] = C12->data[i * halfSize + j];
            result->data[(i + halfSize) * fullSize + j] = C21->data[i * halfSize + j];
            result->data[(i + halfSize) * fullSize + j + halfSize] = C22->data[i * halfSize + j];
        }
    }

    return result;
}

int check_result(Matrix* result_by_naive, Matrix* result_by_ring, double tolerance) {
    if (result_by_naive->rows != result_by_ring->rows || result_by_naive->cols != result_by_ring->cols) {
        return 0;  // row and col do not match
    }

    for (int i = 0; i < result_by_naive->rows; i++) {
        for (int j = 0; j < result_by_naive->cols; j++) {
            double diff = fabs(result_by_naive->data[i * result_by_naive->cols + j] - result_by_ring->data[i * result_by_ring->cols + j]);
            if (diff > tolerance) {
                return 0;  // element does not match
            }
        }
    }

    return 1;  // all elements match
}

int quick_check_result(Matrix* result_1, Matrix* result_2, double tolerance) {
    if (result_1->rows != result_2->rows || result_1->cols != result_2->cols) {
        return 0;  // 行和列不匹配
    }

    int num_checks = result_1->rows;  // 将检查次数设置为矩阵的行数
    int total_elements = result_1->rows * result_1->cols;
    for (int check = 0; check < num_checks; check++) {
        // 随机选取一个元素进行检查
        int index = rand() % total_elements;
        double diff = fabs(result_1->data[index] - result_2->data[index]);
        if (diff > tolerance) {
            return 0;  // 元素不匹配
        }
    }

    return 1;  // 检查的元素匹配
}

void prepare_strassen(Matrix* A, Matrix* B, Matrix* M_A[], Matrix* M_B[]) {
    // 为子矩阵分配空间
    int local_n = A->rows / 2;
    Matrix *A11 = allocate_matrix(local_n, local_n), *A12 = allocate_matrix(local_n, local_n),
           *A21 = allocate_matrix(local_n, local_n), *A22 = allocate_matrix(local_n, local_n);
    Matrix *B11 = allocate_matrix(local_n, local_n), *B12 = allocate_matrix(local_n, local_n),
           *B21 = allocate_matrix(local_n, local_n), *B22 = allocate_matrix(local_n, local_n);

    // 检查是否成功分配内存
    if (!A11 || !A12 || !A21 || !A22 || !B11 || !B12 || !B21 || !B22) {
        // 处理分配失败的情况...
        // 注意：需要适当释放已分配的内存
        return;
    }

    // 将矩阵A和B分割成子矩阵
    split_matrix(A, A11, A12, A21, A22);
    split_matrix(B, B11, B12, B21, B22);

    // 计算Strassen算法的M矩阵
    M_A[0] = matrix_add(A11, A22);        
    M_A[1] = matrix_add(A21, A22);        
    M_A[2] = copy_matrix(A11);                    
    M_A[3] = copy_matrix(A22);                  
    M_A[4] = matrix_add(A11, A12);        
    M_A[5] = matrix_subtract(A21, A11);   
    M_A[6] = matrix_subtract(A12, A22);   

    M_B[0] = matrix_add(B11, B22);
    M_B[1] = copy_matrix(B11);
    M_B[2] = matrix_subtract(B12, B22);
    M_B[3] = matrix_subtract(B21, B11);
    M_B[4] = copy_matrix(B22);
    M_B[5] = matrix_add(B11, B12);
    M_B[6] = matrix_add(B21, B22);

    // 释放不再需要的子矩阵内存
    free_matrix(A11); free_matrix(A12); free_matrix(A21); free_matrix(A22);
    free_matrix(B11); free_matrix(B12); free_matrix(B21); free_matrix(B22);
}

// function to check if the process is leader under current level
int is_leader(int rank, int level) {
    return rank < pow(NUM_TASKS, level - 1);
}

// function to check if the process is a worker under current level
int is_worker(int rank, int level) {
    return rank < pow(NUM_TASKS, level);
}

// function to get the rank of the leader process
int get_leader_rank(int rank) {
    return rank / NUM_TASKS;
}

// function to get the rank of the worker process
int* get_worker_rank(int rank) {
    int* worker_rank = (int*) malloc(sizeof(int) * NUM_TASKS);
    for (int i = 0; i < NUM_TASKS; i++) {
        worker_rank[i] = rank * NUM_TASKS + i;
    }
    return worker_rank;
}

// function to print array
void print_data(int* data, int data_size) {
    for (int i = 0; i < data_size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

void distribute_data(Matrix** M_A, Matrix** M_B, Matrix* A, Matrix* B, int local_n, int level) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Matrix* M_A_0 = NULL;
    Matrix* M_B_0 = NULL;
    if (is_leader(rank, level)) {
        // Leader逻辑
        int* worker_rank = get_worker_rank(rank);
        Matrix* M_A[NUM_TASKS];
        Matrix* M_B[NUM_TASKS];
        prepare_strassen(A, B, M_A, M_B);
        // print all Mi first element, and its level and rank
        
        // leader send data to workers if leader is ROOT

        if (rank == ROOT) {

            // leader send M_A[0] and M_B[0] to itselves
            M_A_0 = copy_matrix(M_A[0]);
            M_B_0 = copy_matrix(M_B[0]);

            // send M_A[1] and M_B[1] to other workers by MPI_Isend
            MPI_Request requests[(NUM_TASKS - 1) * 2];
            int request_count = 0;

            for (int i = 1; i < NUM_TASKS; i++) {
                MPI_Isend(M_A[i]->data, M_A[i]->rows * M_A[i]->cols, MPI_DOUBLE, worker_rank[i], 0, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(M_B[i]->data, M_B[i]->rows * M_B[i]->cols, MPI_DOUBLE, worker_rank[i], 1, MPI_COMM_WORLD, &requests[request_count++]);
                // printf("process %d send M_A[%d] and M_B[%d] to process %d\n", rank, i, i, worker_rank[i]);
            }
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
            
        } else {
            MPI_Request requests[(NUM_TASKS) * 2];
            int request_count = 0;
            
            for (int i = 0; i < NUM_TASKS; i++) {
                MPI_Isend(M_A[i]->data, M_A[i]->rows * M_A[i]->cols, MPI_DOUBLE, worker_rank[i], 0, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(M_B[i]->data, M_B[i]->rows * M_B[i]->cols, MPI_DOUBLE, worker_rank[i], 1, MPI_COMM_WORLD, &requests[request_count++]);
                // printf("!!process %d send M_A[%d][0][0] = %f and M_B[%d][0][0] = %f to process %d\n", rank, i, M_A[i]->data[0], i, M_B[i]->data[0], worker_rank[i]);
            }
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);           
        }
        // free martrix M_A and M_B on leader
        for (int i = 0; i < NUM_TASKS; i++) {
            free_matrix(M_A[i]);
            free_matrix(M_B[i]);
        }
    }

    if (is_worker(rank, level)) {
        // Worker逻辑
        int leader_rank = get_leader_rank(rank);
        // worker receive data only if they are not ROOT
        if (rank != ROOT) {
            // worker receive data from leader
            MPI_Request recv_requests[2];
            int recv_count = 0;

            *M_A = allocate_matrix(local_n, local_n);  // 使用您的函数创建矩阵并分配内存
            *M_B = allocate_matrix(local_n, local_n);

            MPI_Irecv((*M_A)->data, local_n * local_n, MPI_DOUBLE, leader_rank, 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
            MPI_Irecv((*M_B)->data, local_n * local_n, MPI_DOUBLE, leader_rank, 1, MPI_COMM_WORLD, &recv_requests[recv_count++]);

            MPI_Waitall(recv_count, recv_requests, MPI_STATUSES_IGNORE);
            // print M_A and M_B first element, and its level and rank
            // printf("process %d receive M_A[0][0] = %f and M_B[0][0] = %f from process %d\n", rank, (*M_A)->data[0], (*M_B)->data[0], leader_rank);
        } else {
            // worker receive data from itselves
            *M_A = allocate_matrix(local_n, local_n);  // 使用您的函数创建矩阵并分配内存
            *M_B = allocate_matrix(local_n, local_n);

            *M_A = M_A_0;
            *M_B = M_B_0;
            // check if M_A and M_B is NULL
            if (*M_A == NULL || *M_B == NULL) {
                printf("M_A or M_B is NULL\n");
            }               
        }

    }

}

Matrix* assemble_C(Matrix* M[], int local_n) {
    // 根据Strassen算法的结果计算最终的子矩阵C11, C12, C21, C22
    Matrix* C11 = matrix_add(matrix_subtract(matrix_add(M[0], M[3]), M[4]), M[6]);
    Matrix* C12 = matrix_add(M[2], M[4]);
    Matrix* C21 = matrix_add(M[1], M[3]);
    Matrix* C22 = matrix_add(matrix_subtract(matrix_add(M[0], M[2]), M[1]), M[5]);

    // 组合这些子矩阵得到最终结果
    Matrix* C = combine_matrix(C11, C12, C21, C22);

    // 释放中间结果占用的内存
    free_matrix(M[0]); free_matrix(M[1]); free_matrix(M[2]); free_matrix(M[3]);
    free_matrix(M[4]); free_matrix(M[5]); free_matrix(M[6]);
    free_matrix(C11); free_matrix(C12); free_matrix(C21); free_matrix(C22);

    return C;
}

void collect_results(Matrix** M_C, Matrix* C, int local_n, int level) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Matrix* M_C_0 = NULL;

    // worker sends data C to its leader under current level
    if (is_worker(rank, level)) {
        if (rank == ROOT) {
            // worker send data to itselves
            M_C_0 = copy_matrix(C);
            free_matrix(C);
        } else {
            // worker send data to its leader
            int leader_rank = get_leader_rank(rank);
            // send by Isend
            MPI_Request send_request;
            MPI_Isend(C->data, local_n * local_n, MPI_DOUBLE, leader_rank, 0, MPI_COMM_WORLD, &send_request);
            // confirm send complete
            MPI_Status send_status;
            MPI_Wait(&send_request, &send_status);
            free_matrix(C);
        }
    }

    // leader receives data from workers and combine into C
    if (is_leader(rank, level)) {
        // leader receives data from workers
        int* worker_rank = get_worker_rank(rank);

        // allocate memory and receive data from workers by Irecv
        Matrix* M[NUM_TASKS];

        if (rank == ROOT) {
            // leader receive data from itselves
            M[0] = M_C_0;

            // leader receive data from workers
            MPI_Request recv_requests[NUM_TASKS - 1];
            int recv_count = 0;

            for (int i = 1; i < NUM_TASKS; i++) {
                M[i] = allocate_matrix(local_n, local_n);
                MPI_Irecv(M[i]->data, local_n * local_n, MPI_DOUBLE, worker_rank[i], 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
            }
            MPI_Waitall(recv_count, recv_requests, MPI_STATUS_IGNORE);
        } else {
            // leader receive data from workers
            MPI_Request recv_requests[NUM_TASKS];
            int recv_count = 0;

            for (int i = 0; i < NUM_TASKS; i++) {
                M[i] = allocate_matrix(local_n, local_n);
                MPI_Irecv(M[i]->data, local_n * local_n, MPI_DOUBLE, worker_rank[i], 0, MPI_COMM_WORLD, &recv_requests[recv_count++]);
            }
             MPI_Waitall(recv_count, recv_requests, MPI_STATUS_IGNORE);
        }

        // print all Mi first element, and its level and rank
        for (int i = 0; i < NUM_TASKS; i++) {
            // printf("level %d: process %d receive M%d[0][0] = %f from process %d\n", level, rank, i+1, M[i]->data[0], worker_rank[i]);
        }
        // compute M into C11 C12 C21 C22 and combine into M_C
        *M_C = assemble_C(M, local_n);
    }

}

Matrix* strassen_multiply_parallel(Matrix* A, Matrix* B, int N, int max_level) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int local_n = N / 2;
    int level;

    // Step 1: divide matrix to workers, until level = max_level
    for (level = 1; level <= max_level; level++) {
        Matrix *M_A = NULL, *M_B = NULL;
        distribute_data(&M_A, &M_B, A, B, local_n, level);
        A = M_A;
        B = M_B;
        if (level < max_level) {
            local_n = local_n / 2; // don't update local_n on the last level
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    level = max_level; // set level to max_level


    // Step 2: compute C = A * B
    // check if A B is NULL
    if (A == NULL || B == NULL) {
        printf("A or B is NULL\n");
    }
    //print matrix A and B first element and its rank
    // printf("process %d: A[0][0] = %f\n", rank, A->data[0]);
    // printf("process %d: B[0][0] = %f\n", rank, B->data[0]);

    Matrix* C = matrix_multiply(A, B);
    // print C's first element and its rank
    free_matrix(A);
    free_matrix(B);

    MPI_Barrier(MPI_COMM_WORLD);
    // Step 3: collect results from workers and combine into C, until level = 1
    for (level = max_level; level >= 1; level--) {
        Matrix *M_C = NULL;
        collect_results(&M_C, C, local_n, level);
        // updata C with M_C
        C = M_C;
        if (level > 1) {
            local_n = local_n * 2; // don't update local_n on the last level
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == ROOT) {
        return C; 
    } else {
        return NULL;
    }
}


Matrix* strassen_multiply_serial(const Matrix* A, const Matrix* B) {
    // 确保输入的矩阵都是方阵且大小相等
    if (A->rows != A->cols || B->rows != B->cols || A->rows != B->rows) {
        return NULL;
    }

    int n = A->rows;
    if (n % 2 != 0 && n != 1) {
        // Strassen算法需要矩阵的维数是2的幂，如果不是，则需要填充矩阵。
        // 这里不实现填充，所以如果n不是2的幂，则直接返回NULL
        return NULL;
    }

    // 为子矩阵分配空间
    int halfSize = n / 2;
    Matrix* A11 = allocate_matrix(halfSize, halfSize);
    Matrix* A12 = allocate_matrix(halfSize, halfSize);
    Matrix* A21 = allocate_matrix(halfSize, halfSize);
    Matrix* A22 = allocate_matrix(halfSize, halfSize);
    Matrix* B11 = allocate_matrix(halfSize, halfSize);
    Matrix* B12 = allocate_matrix(halfSize, halfSize);
    Matrix* B21 = allocate_matrix(halfSize, halfSize);
    Matrix* B22 = allocate_matrix(halfSize, halfSize);

    // 检查是否成功分配内存
    if (!A11 || !A12 || !A21 || !A22 || !B11 || !B12 || !B21 || !B22) {
        // 处理分配失败的情况...
    }

    // 将矩阵A和B分割成子矩阵
    split_matrix(A, A11, A12, A21, A22);
    split_matrix(B, B11, B12, B21, B22);

    // 使用Strassen算法计算M1到M7
    Matrix *M1 = matrix_multiply(matrix_add(A11, A22), matrix_add(B11, B22)); // M1 = (A11 + A22) * (B11 + B22)
    Matrix *M2 = matrix_multiply(matrix_add(A21, A22), B11);                   // M2 = (A21 + A22) * B11
    Matrix *M3 = matrix_multiply(A11, matrix_subtract(B12, B22));              // M3 = A11 * (B12 - B22)
    Matrix *M4 = matrix_multiply(A22, matrix_subtract(B21, B11));              // M4 = A22 * (B21 - B11)
    Matrix *M5 = matrix_multiply(matrix_add(A11, A12), B22);                   // M5 = (A11 + A12) * B22
    Matrix *M6 = matrix_multiply(matrix_subtract(A21, A11), matrix_add(B11, B12)); // M6 = (A21 - A11) * (B11 + B12)
    Matrix *M7 = matrix_multiply(matrix_subtract(A12, A22), matrix_add(B21, B22)); // M7 = (A12 - A22) * (B21 + B22)

    // print all Mi
    printf("serial result first element: \n");
    printf("M1[0][0] = %f\n", M1->data[0]);
    printf("M2[0][0] = %f\n", M2->data[0]); 
    printf("M3[0][0] = %f\n", M3->data[0]);
    printf("M4[0][0] = %f\n", M4->data[0]);
    printf("M5[0][0] = %f\n", M5->data[0]);
    printf("M6[0][0] = %f\n", M6->data[0]);
    printf("M7[0][0] = %f\n", M7->data[0]);

    // 计算结果子矩阵
    Matrix *C11 = matrix_add(matrix_subtract(matrix_add(M1, M4), M5), M7);
    Matrix *C12 = matrix_add(M3, M5);
    Matrix *C21 = matrix_add(M2, M4);
    Matrix *C22 = matrix_add(matrix_subtract(matrix_add(M1, M3), M2), M6);

    // 组合子矩阵得到最终结果
    Matrix *result = combine_matrix(C11, C12, C21, C22);

    // 释放分配的内存
    free_matrix(A11);
    free_matrix(A12);
    free_matrix(A21);
    free_matrix(A22);
    free_matrix(B11);
    free_matrix(B12);
    free_matrix(B21);
    free_matrix(B22);
    free_matrix(M1);
    free_matrix(M2);
    free_matrix(M3);
    free_matrix(M4);
    free_matrix(M5);
    free_matrix(M6);
    free_matrix(M7);
    free_matrix(C11);
    free_matrix(C12);
    free_matrix(C21);
    free_matrix(C22);

    return result;
}