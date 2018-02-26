// To compile: make cuda
// To run: ./a.out [width] [height] [input_file]

#define BLOCK_SIZE 32

#define GEN_LIMIT 1000

#define CHECK_SIMILARITY
#define SIMILARITY_FREQUENCY 3

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>

#include "../include/timestamp.h"

#define cudaSafeCall(call)                                                \
    {                                                                     \
        cudaError err = call;                                             \
        if (cudaSuccess != err)                                           \
        {                                                                 \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

void perror_exit(const char *message)
{
    perror(message);
    exit(EXIT_FAILURE);
}

void print_to_file(unsigned char *univ, int width, int height)
{
    FILE *fout = fopen("./cuda_output.out", "w"); // printing the result to a file with
                                                  // 1 or 0 (1 being an alive cell and 0 a dead cell)
    for (int i = 1; i <= width; i++)
    {
        for (int j = 1; j <= height; j++)
        {
            fprintf(fout, univ[i * (width + 2) + j] ? "1" : "0");
        }
        fprintf(fout, "\n");
    }

    fflush(fout);
    fclose(fout);
}

__global__ void halo_rows(unsigned char *univ, int height)
{
    // Copy the actual rows from both sides to the additional halo rows in the array
    int index = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (index <= height)
    {
        univ[(height + 2) * (height + 1) + index] = univ[(height + 2) + index];
        univ[index] = univ[(height + 2) * height + index];
    }
}

__global__ void halo_cols(unsigned char *univ, int width)
{
    // Copy the actual columns from both sides to the additional halo columns in the array
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index <= width + 1)
    {
        univ[index * (width + 2) + width + 1] = univ[index * (width + 2) + 1];
        univ[index * (width + 2)] = univ[index * (width + 2) + width];
    }
}

__global__ void compare(unsigned char *univ, unsigned char *new_univ, long long int size, int *same)
{
    // Use reduction to check if two arrays are the same by adding 1 to a counter for each common cell:
    // If it's (width + 2) * (height + 2), then they are identical
    __shared__ int s_array[BLOCK_SIZE];

    int index = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    s_array[index] = (i < size) ? (int)(univ[i] == new_univ[i]) : 0;

    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (index < i)
        {
            s_array[index] += s_array[index + i];
        }
        __syncthreads();
    }

    if (index == 0)
        atomicAdd(same, s_array[0]);
}

__global__ void empty(unsigned char *temp_univ, long long int size, int *alive)
{
    // Use reduction to check if the array is empty by adding all values in the array:
    // If it's zero, then there is no alive cell in the array
    __shared__ int s_array[BLOCK_SIZE];

    int index = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    s_array[index] = (i < size) ? (int)temp_univ[i] : 0;

    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (index < i)
        {
            s_array[index] += s_array[index + i];
        }
        __syncthreads();
    }

    if (index == 0)
        atomicAdd(alive, s_array[0]);
}

__global__ void evolve(unsigned char *univ, unsigned char *new_univ, int width, int height)
{
    // Generate new generation: keep it in new_univ
    int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int id = iy * (width + 2) + ix;

    int neighbors = 0;

    if (iy <= height && ix <= width)
    {
        neighbors = univ[id + (height + 2)] +                           // Upper neighbor
                    univ[id - (height + 2)] +                           // Lower neighbor
                    univ[id + 1] +                                      // Right neighbor
                    univ[id - 1] +                                      // Left neighbor
                    univ[id + (height + 3)] + univ[id - (height + 3)] + // Diagonal neighbors
                    univ[id - (height + 1)] + univ[id + (height + 1)];

        new_univ[id] = (neighbors == 3 || (neighbors == 2 && univ[id]));
    }
}

int main(int argc, char *argv[])
{
    int width = 0, height = 0;
    long long int size = 0;

    width = atoi(argv[1]);
    height = atoi(argv[2]);
    size = (width + 2) * (height + 2);

    // Allocate space for the game array
    unsigned char *univ = (unsigned char *)calloc(size, sizeof(unsigned char));
    if (univ == NULL)
        perror_exit("calloc: ");

    // Fetch all the values from an input file
    FILE *filePtr = fopen(argv[3], "r");
    if (filePtr == NULL)
        perror_exit("fopen: ");

    for (int i = 1; i <= width; i++)
    {
        for (int j = 1; j <= height;)
        {
            char c = fgetc(filePtr);
            if ((c != EOF) && (c != '\n'))
            {
                univ[i * (width + 2) + j] = c - 48;
                j++;
            }
        }
    }
    fclose(filePtr);
    filePtr = NULL;

    // Allocate two game arrays in the GPU memory and transfer the contents of the original array
    unsigned char *d_univ, *d_new_univ;

    cudaSafeCall(cudaMalloc((void **)&d_univ, size * sizeof(unsigned char)));
    cudaSafeCall(cudaMalloc((void **)&d_new_univ, size * sizeof(unsigned char)));

    cudaSafeCall(cudaMemcpy(d_univ, univ, size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_new_univ, univ, size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Allocate space in the GPU memory for two counters/flags: alive to check is the array is
    // empty and the same to compare two generations
    int alive = 0, *d_alive,
        same = 0, *d_same;

    cudaSafeCall(cudaMalloc((void **)&d_alive, sizeof(int)));
    cudaSafeCall(cudaMalloc((void **)&d_same, sizeof(int)));

    cudaSafeCall(cudaMemcpy(d_alive, &alive, sizeof(int), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_same, &same, sizeof(int), cudaMemcpyHostToDevice));

    // Set up parameters for number of blocks and threads per block for kernel launches
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    int linGrid = (int)ceil(width / (float)BLOCK_SIZE);
    dim3 grid_size(linGrid, linGrid, 1);

    int rows_grid_size = (int)ceil(width / (float)BLOCK_SIZE);
    int cols_grid_size = (int)ceil((width + 2) / (float)BLOCK_SIZE);
    int univ_grid_size = (width <= BLOCK_SIZE * 2 || height <= BLOCK_SIZE) ? (int)ceil((width + 2) * (height + 2) / (float)BLOCK_SIZE) : (int)ceil((width + 2) * (height + 2) / (float)(BLOCK_SIZE * BLOCK_SIZE));

    int generation = 0;
#ifdef CHECK_SIMILARITY
    int counter = 0;
#endif

    // Get currect timestamp: calculations are about to start
    timestamp t_start;
    t_start = getTimestamp();

    while (generation < GEN_LIMIT)
    {
        // Create halo rows and columns for neighbors
        halo_rows<<<rows_grid_size, BLOCK_SIZE>>>(d_univ, height);
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());

        halo_cols<<<cols_grid_size, BLOCK_SIZE>>>(d_univ, width);
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());

        // Generate a new generations from d_univ and store in on d_new_univ
        evolve<<<grid_size, block_size>>>(d_univ, d_new_univ, width, height);
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());

#ifdef CHECK_SIMILARITY
        // Check new and old generation, if they are the same: exit
        counter++;
        if (counter == SIMILARITY_FREQUENCY)
        {
            compare<<<univ_grid_size, BLOCK_SIZE>>>(d_univ, d_new_univ, size, d_same);
            cudaSafeCall(cudaGetLastError());
            cudaSafeCall(cudaDeviceSynchronize());

            cudaSafeCall(cudaMemcpy(&same, d_same, sizeof(int), cudaMemcpyDeviceToHost));

            if (same == size)
                break;
            else
                cudaSafeCall(cudaMemset(d_same, 0, sizeof(int)));

            counter = 0;
        }
#endif

        // Check if the array is empty, if it is: exit
        empty<<<univ_grid_size, BLOCK_SIZE>>>(d_new_univ, size, d_alive);
        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());

        cudaSafeCall(cudaMemcpy(&alive, d_alive, sizeof(int), cudaMemcpyDeviceToHost));

        if (alive == 0)
            break;
        else
            cudaSafeCall(cudaMemset(d_alive, 0, sizeof(int)));

        // Pointer switch for fast array switching: d_univ will be used in the next loop
        unsigned char *temp_univ = d_univ;
        d_univ = d_new_univ;
        d_new_univ = temp_univ;

        generation++;
    }

    // Get the total duration of the loop above in milliseconds
    float msecs = getElapsedtime(t_start);

    cudaSafeCall(cudaMemcpy(univ, d_univ, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    print_to_file(univ, width, height);

    // Free allocated memory
    cudaSafeCall(cudaFree(d_univ));
    cudaSafeCall(cudaFree(d_new_univ));

    cudaSafeCall(cudaFree(d_alive));
    cudaSafeCall(cudaFree(d_same));

    free(univ);
    univ = NULL;

    printf("Generations:\t%d\n", generation);
    printf("Execution time:\t%.2f msecs\n", msecs);

    printf("Finished\n");
    fflush(stdout);

    return 0;
}
