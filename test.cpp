#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.141592653

#ifdef USE_HIP
#include "hip/hip_runtime.h"

// GPU  kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    a[id] = PI;
    b[id] = PI*PI;
    // Make sure we do not go out of bounds
    if (id < n) c[id] = PI*a[id] + PI*b[id];
}
#endif

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    constexpr int N = 1024;
 
    // Host input vectors
    double *a;
    double *b;
    //Host output vector
    double *c;

    // Size, in bytes, of each vector
    size_t bytes = N*sizeof(double);
 
    // Allocate memory for each vector on host
    a = (double*)malloc(bytes);
    b = (double*)malloc(bytes);
    c = (double*)malloc(bytes);
    
#ifdef USE_HIP
    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
 
    // Allocate memory for each vector on GPU
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)N/blockSize);
 
    // Execute the kernel
    hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(blockSize), 0, 0, d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    // Copy array back to host
    hipMemcpy(a, d_a, bytes, hipMemcpyDeviceToHost);
    hipMemcpy(b, d_b, bytes, hipMemcpyDeviceToHost);
    hipMemcpy(c, d_c, bytes, hipMemcpyDeviceToHost);
 
    // Release device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
#else 
#ifdef USE_OPENMP
    #pragma omp parallel for
    for(int i = 0; i < N; i++ ) {
        a[i] = PI;
        b[i] = PI*PI;
        c[i] = PI*a[i] + PI*b[i];
    }
#else
    #error Unsupporrted backend
#endif
#endif
    // Release host memory
    free(a);
    free(b);
    free(c);

    // Print off a hello world message
    printf("Hello world from rank %d out of %d processors\n", world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
 
    return 0;
}
