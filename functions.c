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
