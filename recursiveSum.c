#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// 函数：递归求和
void recursiveSum(int *array, int left, int right, int level, int max_level, int rank, int numprocs) {
    if (level == max_level) {
        // 执行简单的加总
        int sum = 0;
        for (int i = left; i <= right; i++) {
            sum += array[i];
        }
        // 将结果发送回上一级
        if (level > 0) {
            MPI_Send(&sum, 1, MPI_INT, rank / 2, 0, MPI_COMM_WORLD);
        } else {
            // 如果是根节点，打印结果
            printf("Total Sum: %d\n", sum);
        }
    } else {
        // 分发任务给子节点
        int mid = (left + right) / 2;
        if (rank * 2 + 1 < numprocs) {
            MPI_Send(&array[left], mid - left + 1, MPI_INT, rank * 2 + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&array[mid + 1], right - mid, MPI_INT, rank * 2 + 2, 0, MPI_COMM_WORLD);

            int sum1, sum2;
            MPI_Recv(&sum1, 1, MPI_INT, rank * 2 + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&sum2, 1, MPI_INT, rank * 2 + 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int total = sum1 + sum2;
            if (level > 0) {
                MPI_Send(&total, 1, MPI_INT, rank / 2, 0, MPI_COMM_WORLD);
            } else {
                printf("Total Sum: %d\n", total);
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // 假设我们有一个固定大小的数组
    int arraySize = 16;
    int *array = (int*)malloc(arraySize * sizeof(int));
    // 初始化数组
    for (int i = 0; i < arraySize; i++) {
        array[i] = i + 1; // 数组元素为1, 2, 3, ..., arraySize
    }

    // 设置最大层级
    int max_level = 3; // 根据你的core结构调整

    // 开始递归求和
    recursiveSum(array, 0, arraySize - 1, 0, max_level, rank, numprocs);

    free(array);
    MPI_Finalize();
    return 0;
}
