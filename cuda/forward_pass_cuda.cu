#include "cuda_runtime.h"

__global__ void batch_tensordot(int * x_in, float * w, float * v, int batchSize, int N_in, int N_out, int t_max)
{
    int index = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    if (index < batch_size && row < N_out && col < t_max){

    for (int j = 0; j < N_in; j++)
    {
	    sum += x_in[index*N_in*t_max + j*t_max + col] * w[j*N_in + row];
    }
    v[index*N_out*t_max + row*t_max + col] = sum;
}

}


__global__ void batch_cumsum(float * v, int N_out, int t_max, int batchSize)
{
    int index = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < batchSize && row < N_out && col == 0){

    for (int j = 1; j < t_max; j++){
            v[index*N_out*t_max + row*t_max + j] += v[index*N_out*t_max + row*t_max + j-1];
    }
    }
}

__global__ void batch_thresholding(int batchSize, int * x, int * firing_t, float * v, int N_out, int t_max, float th_val)
{
	int index = blockIdx.z * blockDim.z + threadIdx.z;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < batchSize && row < N_out && col == 0){
		
	for (int j = 0; j < t_max; j++){
           if (v[index*N_out*t_max + row*t_max + j] >= th_val){
              firing_t[index*N_out + row] = j;
              x[index*N_out*t_max + row*t_max + j] = 1;
              break;
              
           }
		   
	           
	}

}
}

void batch_dense(int * x_in, int * x_out, int * firing_t, float * w, int N_in, int N_out, int batchSize, float th_val, int t_max){
    float * v;
    cudaMalloc((void **) &v, sizeof(float)*batchSize*N_out*t_max);

    dim3 threadsPerBlock(t_max, N_out, batchSize);
    dim3 blocksPerGrid(1, 1, 1);

    if (batchSize*N_out*t_max > 1024){
        threadsPerBlock.x = 1024;
        threadsPerBlock.y = 1024;
        threadsPerBlock.z = 1024;
        blocksPerGrid.z = ceil(float(batchSize)/float(threadsPerBlock.x));
        blocksPerGrid.y = ceil(float(N_out)/float(threadsPerBlock.y));
        blocksPerGrid.x = ceil(float(t_max)/float(threadsPerBlock.y));
    }
    batch_tensordot<<<blocksPerGrid, threadsPerBlock>>>(x_in, w, v, batchSize, N_in, N_out, t_max);
    batch_cumsum<<<blocksPerGrid, threadsPerBlock>>>(v, N_out, t_max, batchSize);
    batch_thresholding<<<blocksPerGrid, threadsPerBlock>>>(batchSize, x_out, firing_t, v, N_out, t_max, th_val);

}
