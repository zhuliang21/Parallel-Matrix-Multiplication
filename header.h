// header.h
#ifndef HEADER_H
#define HEADER_H

#include <stdlib.h> // For malloc and rand
#include <stdio.h> // For printf
#include <mpi.h> // For MPI functions
#include <string.h> // For memcpy
#include <math.h> 

#define ROOT 0  // set root process
#define NUM_TASKS 7


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

int quick_check_result(Matrix* result_1, Matrix* result_2, double tolerance);

void prepare_strassen(Matrix* A, Matrix* B, Matrix* M_A[], Matrix* M_B[]);

int is_leader(int rank, int level);

int is_worker(int rank, int level);

int get_leader_rank(int rank);

int* get_worker_rank(int rank);

void print_data(int* data, int data_size);

void distribute_data(Matrix** M_A, Matrix** M_B, Matrix* A, Matrix* B, int local_n, int level);

#endif // HEADER_H
 