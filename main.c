#include "header.h"

// int main() {

//     // 创建一个4x4的矩阵并填充随机数
//     Matrix* original_matrix = generate_matrix(4, 4);
//     if (original_matrix == NULL) {
//         fprintf(stderr, "Memory allocation failed for original matrix.\n");
//         return EXIT_FAILURE;
//     }

//     // 打印原始矩阵
//     printf("Original Matrix:\n");
//     print_matrix(original_matrix);

//     // 创建四个子矩阵
//     int sub_size = original_matrix->rows / 2; // 或者 cols，因为矩阵是正方形的
//     Matrix *block11 = allocate_matrix(sub_size, sub_size);
//     Matrix *block12 = allocate_matrix(sub_size, sub_size);
//     Matrix *block21 = allocate_matrix(sub_size, sub_size);
//     Matrix *block22 = allocate_matrix(sub_size, sub_size);

    
//     // 分割矩阵
//     split_matrix(original_matrix, block11, block12, block21, block22);

//     // 打印分割后的子矩阵
//     printf("Block 11:\n");
//     print_matrix(block11);
//     printf("Block 12:\n");
//     print_matrix(block12);
//     printf("Block 21:\n");
//     print_matrix(block21);
//     printf("Block 22:\n");
//     print_matrix(block22);

//     // 释放原始矩阵和子矩阵内存
//     free_matrix(original_matrix);
//     free_matrix(block11);
//     free_matrix(block12);
//     free_matrix(block21);
//     free_matrix(block22);

//     return EXIT_SUCCESS;
// }

// int main() {

//     // 生成两个4x4的矩阵
//     Matrix *A = generate_matrix(2, 2);
//     Matrix *B = generate_matrix(2, 2);
    
//     // 确保矩阵已经成功创建
//     if (A == NULL || B == NULL) {
//         printf("Error creating matrices.\n");
//         free_matrix(A); // 即使为NULL也是安全的
//         free_matrix(B); // 即使为NULL也是安全的
//         return -1;
//     }
    
//     // 打印矩阵A
//     printf("Matrix A:\n");
//     print_matrix(A);

//     // 打印矩阵B
//     printf("\nMatrix B:\n");
//     print_matrix(B);

//     // 相加矩阵A和B
//     Matrix *result = matrix_add(A, B);
    
//     // 检查结果
//     if (result == NULL) {
//         printf("Error adding matrices.\n");
//         free_matrix(A);
//         free_matrix(B);
//         return -1;
//     }

//     // 打印结果矩阵
//     printf("\nResult (A+B):\n");
//     print_matrix(result);

//     // 释放内存
//     free_matrix(A);
//     free_matrix(B);
//     free_matrix(result);

//     return 0;
// }

// int main() {

//     // 生成两个 2x2 的矩阵
//     Matrix *A = generate_matrix(2, 2);
//     Matrix *B = generate_matrix(2, 2);

//     if (A == NULL || B == NULL) {
//         printf("Error creating matrices.\n");
//         free_matrix(A);
//         free_matrix(B);
//         return -1;
//     }
    
//     // 打印矩阵 A 和 B
//     printf("Matrix A:\n");
//     print_matrix(A);
//     printf("\nMatrix B:\n");
//     print_matrix(B);

//     // 减去矩阵 B 从 A
//     Matrix *result = matrix_subtract(A, B);
//     if (result == NULL) {
//         printf("Error subtracting matrices.\n");
//         free_matrix(A);
//         free_matrix(B);
//         return -1;
//     }

//     // 打印结果矩阵
//     printf("\nResult (A-B):\n");
//     print_matrix(result);

//     // 释放内存
//     free_matrix(A);
//     free_matrix(B);
//     free_matrix(result);

//     return 0;
// }

// main 函数来测试 matrix_multiply
// int main() {

//     // 生成两个 4x4 的矩阵
//     Matrix *A = generate_matrix(2, 2);
//     Matrix *B = generate_matrix(2, 2);

//     if (A == NULL || B == NULL) {
//         printf("Error creating matrices.\n");
//         free_matrix(A);
//         free_matrix(B);
//         return -1;
//     }
    
//     // 打印矩阵 A 和 B
//     printf("Matrix A:\n");
//     print_matrix(A);
//     printf("\nMatrix B:\n");
//     print_matrix(B);

//     // 计算矩阵 A 和 B 的乘积
//     Matrix *result = matrix_multiply(A, B);
//     if (result == NULL) {
//         printf("Error multiplying matrices.\n");
//         free_matrix(A);
//         free_matrix(B);
//         return -1;
//     }

//     // 打印结果矩阵
//     printf("\nResult (A*B):\n");
//     print_matrix(result);

//     // 释放内存
//     free_matrix(A);
//     free_matrix(B);
//     free_matrix(result);

//     return 0;
// }

// int main() {

//     // Generate two 4x4 matrices with random values
//     Matrix* A = generate_matrix(4, 4);
//     Matrix* B = generate_matrix(4, 4);

//     // Make sure matrices are generated
//     if (A == NULL || B == NULL) {
//         fprintf(stderr, "Failed to allocate matrices.\n");
//         exit(EXIT_FAILURE);
//     }

//     // Print matrices
//     printf("Matrix A:\n");
//     print_matrix(A);
//     printf("\nMatrix B:\n");
//     print_matrix(B);

//     // Perform Strassen multiplication
//     Matrix* C = strassen_multiply_serial(A, B);

//     // Make sure multiplication was successful
//     if (C == NULL) {
//         fprintf(stderr, "Strassen multiplication failed.\n");
//         exit(EXIT_FAILURE);
//     }

//     // Print result
//     printf("\nMatrix C (Result of A * B):\n");
//     print_matrix(C);

//     // Clean up
//     free_matrix(A);
//     free_matrix(B);
//     free_matrix(C);

//     return 0;
// }

// int main() {


//     // 步骤 1: 生成两个 4x4 的矩阵 A 和 B
//     Matrix* A = generate_matrix(4, 4);
//     Matrix* B = generate_matrix(4, 4);

//     if (A == NULL || B == NULL) {
//         fprintf(stderr, "Failed to allocate matrices.\n");
//         // 如果 A 或 B 分配失败，记得释放已分配的矩阵并退出
//         free_matrix(A);
//         free_matrix(B);
//         exit(EXIT_FAILURE);
//     }

//     // 打印矩阵 A 和 B
//     printf("Matrix A:\n");
//     print_matrix(A);
//     printf("\nMatrix B:\n");
//     print_matrix(B);

//     // 步骤 2: 使用常规矩阵乘法计算 C
//     Matrix* C = matrix_multiply(A, B);
//     printf("\nMatrix C (Regular Multiply):\n");
//     print_matrix(C);

//     // 步骤 3: 使用 Strassen 算法计算 D
//     Matrix* D = strassen_multiply_serial_recursive(A, B, 0, 1);
//     printf("\nMatrix D (Strassen Multiply):\n");
//     print_matrix(D);

//     // 步骤 4: 检查 C 和 D 是否一致
//     double tolerance = 1e-6; // 定义容忍的差异阈值
//     if (check_result(C, D, tolerance)) {
//         printf("\nResult is correct: Matrices C and D are the same within tolerance.\n");
//     } else {
//         printf("\nResult is incorrect: Matrices C and D differ more than tolerance.\n");
//     }

//     // 释放所有分配的内存
//     free_matrix(A);
//     free_matrix(B);
//     free_matrix(C);
//     free_matrix(D);

//     return 0;
// }


int main(int argc, char** argv) {
    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    // print size of MPI_COMM_WORLD
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // 获取当前进程的rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 1 << 10;

    // 创建矩阵A和B，只在根进程进行
    Matrix *A = NULL, *B = NULL;
    if (rank == 0) {
        // generate matrix A and B on 
        A = generate_matrix(N, N);  
        B = generate_matrix(N, N);
        printf("generate matrix A and B on root process\n");
    }

    // 执行Strassen算法 level 1
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    Matrix* C_by_parallel_strassen = strassen_multiply(A, B);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double time = end_time - start_time;

    // 根进程打印结果矩阵C
    if (rank == 0) {
        // printf("Matrix A:\n");
        // print_matrix(A);
        // printf("Matrix B:\n");
        // print_matrix(B);

        printf("Time for strassen_multiply_level_1: %f seconds.\n", time);

        Matrix* C_by_serial_strassen = strassen_multiply_serial(A, B);
        printf("Matrix C by serial_strassen computation is done.\n");
        
        printf("Matrix C by serial_strassen:\n");
        // print the first element of C_by_serial_strassen
        printf("C_by_serial_strassen[0][0] = %f\n", C_by_serial_strassen->data[0]);
        // print the size of C_by_serial_strassen
        printf("C_by_serial_strassen->rows = %d\n", C_by_serial_strassen->rows);
        // print the first element of C_by_parallel_strassen
        printf("C_by_parallel_strassen[0][0] = %f\n", C_by_parallel_strassen->data[0]);
        // print the size of C_by_parallel_strassen
        printf("C_by_parallel_strassen->rows = %d\n", C_by_parallel_strassen->rows);

        free_matrix(A);
        free_matrix(B);

        if (quick_check_result(C_by_parallel_strassen, C_by_serial_strassen, 1e-3) ) {
            printf("Pass Result Check.\n");
        } else {
            printf("Fail Result Check.\n");
        }

        free_matrix(C_by_serial_strassen);
    }

    // 终止MPI环境
    MPI_Finalize();
    return 0;
}
