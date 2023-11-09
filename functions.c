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


// Function to multiply two matrices using Strassen's algorithm on a single process under Level 1 

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

Matrix* strassen_multiply_serial_recursive(const Matrix* A, const Matrix* B, int currentLevel, int maxLevel) {
    // Ensure the input matrices are square and of the same size
    if (A->rows != A->cols || B->rows != B->cols || A->rows != B->rows) {
        return NULL;
    }

    int n = A->rows;
    // Base case for recursion: use standard multiplication when level is maxed out or matrix is small enough
    if ((currentLevel >= maxLevel) || (n <= 2)) {
        return matrix_multiply(A, B);
    }

    if (n % 2 != 0 && n != 1) {
        // Strassen's algorithm requires matrices to have dimensions of a power of two
        // Padding is not implemented here, return NULL if not a power of two
        return NULL;
    }

    // Allocate space for submatrices
    int halfSize = n / 2;
    Matrix* A11 = allocate_matrix(halfSize, halfSize);
    Matrix* A12 = allocate_matrix(halfSize, halfSize);
    Matrix* A21 = allocate_matrix(halfSize, halfSize);
    Matrix* A22 = allocate_matrix(halfSize, halfSize);
    Matrix* B11 = allocate_matrix(halfSize, halfSize);
    Matrix* B12 = allocate_matrix(halfSize, halfSize);
    Matrix* B21 = allocate_matrix(halfSize, halfSize);
    Matrix* B22 = allocate_matrix(halfSize, halfSize);

    // Check for successful memory allocation
    if (!A11 || !A12 || !A21 || !A22 || !B11 || !B12 || !B21 || !B22) {
        // 处理分配失败的情况...
    }

    // Split matrices A and B into submatrices
    split_matrix(A, A11, A12, A21, A22);
    split_matrix(B, B11, B12, B21, B22);

    // Recursively apply Strassen's algorithm to compute M1 through M7 using the new function
    Matrix *M1 = strassen_multiply_serial_recursive(matrix_add(A11, A22), matrix_add(B11, B22), currentLevel + 1, maxLevel);
    Matrix *M2 = strassen_multiply_serial_recursive(matrix_add(A21, A22), B11, currentLevel + 1, maxLevel);
    Matrix *M3 = strassen_multiply_serial_recursive(A11, matrix_subtract(B12, B22), currentLevel + 1, maxLevel);
    Matrix *M4 = strassen_multiply_serial_recursive(A22, matrix_subtract(B21, B11), currentLevel + 1, maxLevel);
    Matrix *M5 = strassen_multiply_serial_recursive(matrix_add(A11, A12), B22, currentLevel + 1, maxLevel);
    Matrix *M6 = strassen_multiply_serial_recursive(matrix_subtract(A21, A11), matrix_add(B11, B12), currentLevel + 1, maxLevel);
    Matrix *M7 = strassen_multiply_serial_recursive(matrix_subtract(A12, A22), matrix_add(B21, B22), currentLevel + 1, maxLevel);


    // Calculate the resulting submatrices C11, C12, C21, C22
    Matrix *C11 = matrix_add(matrix_subtract(matrix_add(M1, M4), M5), M7);
    Matrix *C12 = matrix_add(M3, M5);
    Matrix *C21 = matrix_add(M2, M4);
    Matrix *C22 = matrix_add(matrix_subtract(matrix_add(M1, M3), M2), M6);

    // Combine submatrices to get the final result
    Matrix *result = combine_matrix(C11, C12, C21, C22);

    // Free allocated memory
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

// Function to check the result
int check_result(Matrix* result_1, Matrix* result_2, double tolerance) {
    if (result_1->rows != result_2->rows || result_1->cols != result_2->cols) {
        return 0;  // row and col do not match
    }

    for (int i = 0; i < result_1->rows; i++) {
        for (int j = 0; j < result_1->cols; j++) {
            double diff = fabs(result_1->data[i * result_1->cols + j] - result_2->data[i * result_2->cols + j]);
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


Matrix* strassen_multiply_level_1(Matrix* A, Matrix* B) {
    // 初始化进程相关信息
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 确保有足够的进程来执行7次并行乘法
    if (size != 7) {
        // 报错或者退出
        fprintf(stderr, "This function requires exactly 7 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Matrix *C = NULL;
    int local_n;

    // on ROOT process
    if (rank == 0) {
        // calculate local_n on root process
        int N = A->rows;
        local_n = N / 2;

        // 为子矩阵分配空间 on root process
        Matrix* A11 = allocate_matrix(local_n, local_n); Matrix* A12 = allocate_matrix(local_n, local_n);
        Matrix* A21 = allocate_matrix(local_n, local_n); Matrix* A22 = allocate_matrix(local_n, local_n);
        Matrix* B11 = allocate_matrix(local_n, local_n); Matrix* B12 = allocate_matrix(local_n, local_n);
        Matrix* B21 = allocate_matrix(local_n, local_n); Matrix* B22 = allocate_matrix(local_n, local_n);

        // 检查是否成功分配内存
        if (!A11 || !A12 || !A21 || !A22 || !B11 || !B12 || !B21 || !B22) {
            // 处理分配失败的情况...
            return NULL;
        }
        printf("ROOT Process %d allocated all submatrix\n", rank);

        // 将矩阵A和B分割成子矩阵
        split_matrix(A, A11, A12, A21, A22);
        split_matrix(B, B11, B12, B21, B22);

        Matrix* M1_A = matrix_add(A11, A22);        
        Matrix* M1_B = matrix_add(B11, B22);
        Matrix* M2_A = matrix_add(A21, A22);        
        Matrix* M2_B = copy_matrix(B11);
        Matrix* M3_A = copy_matrix(A11);                    
        Matrix* M3_B = matrix_subtract(B12, B22);
        Matrix* M4_A = copy_matrix(A22);                  
        Matrix* M4_B = matrix_subtract(B21, B11);
        Matrix* M5_A = matrix_add(A11, A12);        
        Matrix* M5_B = copy_matrix(B22);
        Matrix* M6_A = matrix_subtract(A21, A11);   
        Matrix* M6_B = matrix_add(B11, B12);
        Matrix* M7_A = matrix_subtract(A12, A22);   
        Matrix* M7_B = matrix_add(B21, B22);

        free_matrix(A11); free_matrix(A12); free_matrix(A21); free_matrix(A22);
        free_matrix(B11); free_matrix(B12); free_matrix(B21); free_matrix(B22);

        // store all Mi_A and Mi_B into array
        Matrix* M_A[7] = {M1_A, M2_A, M3_A, M4_A, M5_A, M6_A, M7_A};
        Matrix* M_B[7] = {M1_B, M2_B, M3_B, M4_B, M5_B, M6_B, M7_B};
        printf("ROOT Process %d calculated all Mi_A and Mi_B\n", rank);

        MPI_Request requests[NUM_TASKS * 2]; // 存储非阻塞通信的请求对象
        int request_count = 0;
        printf("ROOT Process %d start to send all Mi_A and Mi_B\n", rank);

        for (int i = 0; i < size; ++i) {
            // 发送 Mi_A
            MPI_Isend(M_A[i]->data, local_n * local_n, MPI_DOUBLE, i, TAG_Mi_A(i), MPI_COMM_WORLD, &requests[request_count++]);
            // printf("ROOT Process %d sent Mi_A to process %d\n", rank, i);
            // 发送 Mi_B
            MPI_Isend(M_B[i]->data, local_n * local_n, MPI_DOUBLE, i, TAG_Mi_B(i), MPI_COMM_WORLD, &requests[request_count++]);
            // printf("ROOT Process %d sent Mi_B to process %d\n", rank, i);
        }
        
        printf("ROOT Process %d sent ALL Mi_A and Mi_B\n", rank);

    } 

    // boardcast local_n to all processes from ROOT
    MPI_Bcast(&local_n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    printf("Process %d received local_n = %d\n", rank, local_n);

    // 为子矩阵分配空间 on all processes
    double *local_A = (double*) malloc(local_n * local_n * sizeof(double));
    double *local_B = (double*) malloc(local_n * local_n * sizeof(double));
    printf("Process %d allocated local_A and local_B\n", rank);
    Matrix* M_A = malloc(sizeof(Matrix));
    Matrix* M_B = malloc(sizeof(Matrix));

    printf("Process %d allocated M_A and M_B\n", rank);

    // 在所有进程上接收 Mi_A 和 Mi_B
    MPI_Request recv_requests[2];
    int recv_count = 0;

    // 接收 Mi_A
    MPI_Irecv(local_A, local_n * local_n, MPI_DOUBLE, ROOT, TAG_Mi_A(rank), MPI_COMM_WORLD, &recv_requests[recv_count++]);
    // 接收 Mi_B
    MPI_Irecv(local_B, local_n * local_n, MPI_DOUBLE, ROOT, TAG_Mi_B(rank), MPI_COMM_WORLD, &recv_requests[recv_count++]);
    // 等待所有非阻塞通信完成
    MPI_Waitall(recv_count, recv_requests, MPI_STATUS_IGNORE);

    // build matrix M_A
    M_A -> data = local_A; M_A -> rows = local_n; M_A -> cols = local_n;
    // build matrix M_B
    M_B -> data = local_B; M_B -> rows = local_n; M_B -> cols = local_n;

    printf("Woker process %d received Mi_A and Mi_B\n", rank);
    // calculate local_result M = M_A * M_B
    Matrix* M = matrix_multiply(M_A, M_B);

    printf("Woker process %d calculated local_result M\n", rank);
    
    // print the first element of M
    printf("M[0][0] = %f\n", M->data[0]);

    // send local_result M to ROOT
    MPI_Request send_requests;
    MPI_Isend(M->data, local_n * local_n, MPI_DOUBLE, ROOT, TAG_Mi(rank), MPI_COMM_WORLD, &send_requests);
    printf("Woker process %d sent local_result M to ROOT\n", rank);

    // 释放分配的内存
    free_matrix(M_A);
    free_matrix(M_B);
    free_matrix(M);

    // on ROOT process
    if (rank == ROOT){
        double* local_results[NUM_TASKS];
        MPI_Request recv_requests[NUM_TASKS];
        int recv_count = 0;

        for (int i = 0; i < NUM_TASKS; ++i) {
            local_results[i] = malloc(local_n * local_n * sizeof(double));
            MPI_Irecv(local_results[i], local_n * local_n, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_Mi(i), MPI_COMM_WORLD, &recv_requests[recv_count++]);
        }
        for (int i = 0; i < recv_count; ++i) {
            MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
        }
        // 等待所有非阻塞通信完成
        MPI_Waitall(recv_count, recv_requests, MPI_STATUS_IGNORE);

        
        printf("ROOT Process %d received all local_results\n", rank);

        // 为Mi分配空间 on root process
        Matrix* M[NUM_TASKS];
        for (int i = 0; i < NUM_TASKS; ++i) {
            M[i] = malloc(sizeof(Matrix));
            M[i]->data = local_results[i];
            M[i]->rows = local_n;
            M[i]->cols = local_n;
        }
        // print all Mi
        for (int i = 0; i < NUM_TASKS; ++i) {
            // printf("ROOT Process %d received Mi from process %d\n", rank, i);
            // print the first element of M
            printf("M%d[0][0] = %f\n", i + 1, M[i]->data[0]);
        }

        Matrix* C11 = matrix_add(matrix_subtract(matrix_add(M[0], M[3]), M[4]), M[6]);
        Matrix* C12 = matrix_add(M[2], M[4]);
        Matrix* C21 = matrix_add(M[1], M[3]);
        Matrix* C22 = matrix_add(matrix_subtract(matrix_add(M[0], M[2]), M[1]), M[5]);
        printf("ROOT Process %d calculated C11, C12, C21, C22\n", rank);

        free_matrix(M[0]); free_matrix(M[1]); free_matrix(M[2]); free_matrix(M[3]);
        free_matrix(M[4]); free_matrix(M[5]); free_matrix(M[6]);

        // 组合子矩阵得到最终结果
        C = combine_matrix(C11, C12, C21, C22);

        free_matrix(C11); free_matrix(C12); free_matrix(C21); free_matrix(C22);
    }
    return C;
}

void worker_receive_submatrices(int rank, int local_n, Matrix** M_A, Matrix** M_B) {
    // 为子矩阵分配空间
    double *local_A = (double*) malloc(local_n * local_n * sizeof(double));
    double *local_B = (double*) malloc(local_n * local_n * sizeof(double));
    printf("Process %d allocated local_A and local_B\n", rank);

    // 在所有进程上接收 Mi_A 和 Mi_B
    MPI_Request recv_requests[2];
    int recv_count = 0;

    // 接收 Mi_A
    MPI_Irecv(local_A, local_n * local_n, MPI_DOUBLE, ROOT, TAG_Mi_A(rank), MPI_COMM_WORLD, &recv_requests[recv_count++]);
    // 接收 Mi_B
    MPI_Irecv(local_B, local_n * local_n, MPI_DOUBLE, ROOT, TAG_Mi_B(rank), MPI_COMM_WORLD, &recv_requests[recv_count++]);
    // 等待所有非阻塞通信完成
    MPI_Waitall(recv_count, recv_requests, MPI_STATUS_IGNORE);

    // 分配和构建矩阵结构
    *M_A = malloc(sizeof(Matrix));
    (*M_A)->data = local_A; 
    (*M_A)->rows = local_n; 
    (*M_A)->cols = local_n;

    *M_B = malloc(sizeof(Matrix));
    (*M_B)->data = local_B; 
    (*M_B)->rows = local_n; 
    (*M_B)->cols = local_n;
}

void root_receive_results(Matrix* M[], int local_n) {
    double* local_results[NUM_TASKS];
    MPI_Request recv_requests[NUM_TASKS];
    int recv_count = 0;

    // 接收来自worker进程的结果
    for (int i = 0; i < NUM_TASKS; ++i) {
        local_results[i] = malloc(local_n * local_n * sizeof(double));
        MPI_Irecv(local_results[i], local_n * local_n, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_Mi(i), MPI_COMM_WORLD, &recv_requests[recv_count++]);
    }

    // 等待所有非阻塞通信完成
    MPI_Waitall(recv_count, recv_requests, MPI_STATUS_IGNORE);
    printf("ROOT received all local results\n");

    // 为每个结果创建Matrix结构
    for (int i = 0; i < NUM_TASKS; ++i) {
        M[i] = malloc(sizeof(Matrix));
        M[i]->data = local_results[i];
        M[i]->rows = local_n;
        M[i]->cols = local_n;
    }
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

void root_send_submatrices(Matrix* M_A[], Matrix* M_B[], int local_n) {
    MPI_Request requests[NUM_TASKS * 2];
    int request_count = 0;
    printf("ROOT Process starts to send all Mi_A and Mi_B\n");

    for (int i = 0; i < NUM_TASKS; ++i) {
        // 发送 Mi_A
        MPI_Isend(M_A[i]->data, local_n * local_n, MPI_DOUBLE, i, TAG_Mi_A(i), MPI_COMM_WORLD, &requests[request_count++]);
        // printf("ROOT sent Mi_A to process %d\n", i);

        // 发送 Mi_B
        MPI_Isend(M_B[i]->data, local_n * local_n, MPI_DOUBLE, i, TAG_Mi_B(i), MPI_COMM_WORLD, &requests[request_count++]);
        // printf("ROOT sent Mi_B to process %d\n", i);
    }

    printf("ROOT sent ALL Mi_A and Mi_B\n");
}

void worker_send_results(Matrix* M, int rank, int local_n) {
    // 发送local_result M到ROOT
    MPI_Request send_request;
    MPI_Isend(M->data, local_n * local_n, MPI_DOUBLE, ROOT, TAG_Mi(rank), MPI_COMM_WORLD, &send_request);
    printf("Worker process %d sent local_result M to ROOT\n", rank);
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


Matrix* strassen_multiply(Matrix* A, Matrix* B) {
    // 初始化进程相关信息
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 确保有足够的进程来执行7次并行乘法
    if (size != 7) {
        // 报错或者退出
        fprintf(stderr, "This function requires exactly 7 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Matrix *C = NULL;
    int local_n;

    // on ROOT process
    if (rank == 0) {
        // calculate local_n on root process
        int N = A->rows;
        local_n = N / 2;

        // 为子矩阵分配空间 on root process
        Matrix* M_A[7];  // 声明一个包含7个Matrix指针的数组
        Matrix* M_B[7];  // 同样，为B矩阵的子矩阵准备一个数组

        // 调用strassen_preprocess函数之前，这些数组中的元素可以初始化为NULL
        for (int i = 0; i < 7; ++i) {
            M_A[i] = NULL;
            M_B[i] = NULL;
        }        
        prepare_strassen(A, B, M_A, M_B);
        printf("ROOT Process %d calculated all Mi_A and Mi_B\n", rank);

        root_send_submatrices(M_A, M_B, local_n);
    } 

    // boardcast local_n to all processes from ROOT
    MPI_Bcast(&local_n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    printf("Process %d received local_n = %d\n", rank, local_n);

    Matrix *M_A = NULL, *M_B = NULL;
    worker_receive_submatrices(rank, local_n, &M_A, &M_B);

    printf("Woker process %d received Mi_A and Mi_B\n", rank);
    // calculate local_result M = M_A * M_B
    Matrix* M = matrix_multiply(M_A, M_B);

    printf("Woker process %d calculated local_result M\n", rank);
    
    // print the first element of M
    printf("M[0][0] = %f\n", M->data[0]);

    worker_send_results(M, rank, local_n);
    // 释放分配的内存
    free_matrix(M_A);
    free_matrix(M_B);
    free_matrix(M);

    // on ROOT process
    if (rank == ROOT){
        Matrix* M[NUM_TASKS];
        root_receive_results(M, local_n);
        // print all Mi
        for (int i = 0; i < NUM_TASKS; ++i) {
            // printf("ROOT Process %d received Mi from process %d\n", rank, i);
            // print the first element of M
            printf("M%d[0][0] = %f\n", i + 1, M[i]->data[0]);
        }
        // 组合M矩阵得到结果C矩阵
        C = assemble_C(M, local_n);
    }
    return C;
}

