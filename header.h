// header.h
#ifndef HEADER_H
#define HEADER_H

#include <stdlib.h> // For malloc and rand
#include <stdio.h> // For printf
#include <mpi.h> // For MPI functions
#include <string.h> // For memcpy
#include <math.h> 

#define ROOT 0  // set root process


typedef struct {
    double *data;    // Continuous 1D array for matrix data
    int rows;        // Number of rows in the matrix
    int cols;        // Number of columns in the matrix
} Matrix;

// Function prototypes

int get_n_from_size(int N);

Matrix* allocate_matrix(int rows, int cols);

Matrix* generate_matrix(int rows, int cols);

void free_matrix(Matrix* matrix);

void print_matrix(Matrix* matrix);

void split_matrix(const Matrix *original, Matrix *block11, Matrix *block12, Matrix *block21, Matrix *block22);

Matrix* matrix_add(const Matrix* A, const Matrix* B);

Matrix* matrix_subtract(const Matrix* A, const Matrix* B);

Matrix* matrix_multiply(const Matrix* A, const Matrix* B);

Matrix* combine_matrix(Matrix *C11, Matrix *C12, Matrix *C21, Matrix *C22);

Matrix* strassen_multiply_serial(const Matrix* A, const Matrix* B);

Matrix* strassen_multiply_serial_recursive(const Matrix* A, const Matrix* B, int currentLevel, int maxLevel);

int check_result(Matrix* result_1, Matrix* result_2, double tolerance);

int quick_check_result(Matrix* result_1, Matrix* result_2, double tolerance);

Matrix* strassen_multiply_level_1(Matrix* A, Matrix* B);

Matrix* strassen_multiply(Matrix* A, Matrix* B);

void prepare_strassen(Matrix* A, Matrix* B, Matrix* M_A[], Matrix* M_B[]);

void worker_receive_submatrices(int rank, int local_n, Matrix** M_A, Matrix** M_B);

void root_send_submatrices(Matrix* M_A[], Matrix* M_B[], int local_n);

void root_receive_results(Matrix* M[], int local_n);

void worker_send_results(Matrix* M, int rank, int local_n);


#define NUM_TASKS 7  

#define TAG_BASE_A 111
#define TAG_BASE_B 112
#define TAG_BASE_C 113

#define TAG_Mi_A(i) (TAG_BASE_A + (i) * 10)
#define TAG_Mi_B(i) (TAG_BASE_B + (i) * 10)
#define TAG_Mi(i)  (TAG_BASE_C + (i) * 10)

#endif // HEADER_H
 