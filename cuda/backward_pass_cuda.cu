// __device__ int min(int a, int b){
    // return a <= b ? a : b;
// }

__global__ void loss_calc(int * firing_t, int * y_batch,double * delta, int batchSize, int N, int t_max, int * correct){

    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int min_time[100];
    __shared__ int target[1000];
    __shared__ int winner[100];

    if (index < batchSize && row < N)
    {
        target[index*N + row] = firing_t[index*N + row];
        __syncthreads();
        if (row == 0){
            winner[index] = 0;
            min_time[index] = t_max;

            for (int i = 0; i < N; i++)
            {
                int time = firing_t[index*N + i];
                // printf("%d\n", time);
                winner[index] = time < min_time[index] ? i : winner[index];
                min_time[index] = time < min_time[index] ? time : min_time[index];
                // printf("%d\n", min_time[index]);
            }

            // printf("%d\n", y_batch[index]);
            // printf("Okay0");


            if(winner[index]==y_batch[index]) atomicAdd(correct, 1);
            // printf("%d\n", *correct);
            // printf("Okay");
        }
        __syncthreads();
        // printf("Yes\n");


        if(min_time[index] == t_max)
        {
            // printf("No\n");
            target[index*N + row] = t_max;

            if (row == y_batch[index])
            {
                target[index*N + row] = t_max - gamma_;
            }
        }
        else
        {
            // target[index*N + row] = firing_t[index*N + row];
            target[index*N + row] = (firing_t[index*N + row] - min_time[index]) < gamma_ ? (min(min_time[index] + gamma_, t_max)) : firing_t[index*N + row];
            // printf("%d\n", y_batch[index]);
            if (row == y_batch[index])
            {
                target[index*N + row] = min_time[index];
            }
        }
        // __syncthreads();

        delta[index * N + row] = double(target[index * N + row] - firing_t[index * N + row]) / double(t_max); //delta = target - firing_t;
        // printf("%d\t%d\t%f\n", index, row, delta[index*N + row]);
    }
}

__global__ void compute_norm(double * delta, double * norm, int batchSize, int N){

    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < batchSize && row < N && threadIdx.x == 0){
        double sum = 0;
        for(int i = 0; i < blockDim.x; i++){ 
            sum += delta[index*N + i]*delta[index*N + i];

        atomicAdd(&norm[index], sum);
    }
}
}

__global__ void grad_norm(double * delta, double * norm, int batchSize, int N){

    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    norm[index] = sqrt(norm[index]);

    if (index < batchSize && row < N){
        delta[index*N + row] /= norm[index];
    }
}

__global__ void batch_bcast_dot(double * delta, bool * fired_before, double * dw_batch, int batchSize, int N_in, int N_out){

    int index = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < batchSize && row < N_in && col < N_out)
    {
        // printf("%lf\n", delta[index*N_out + col]);
        dw_batch[index*N_in*N_out + row*N_out + col] = delta[index*N_out + col]*fired_before[index*N_in*N_out + row*N_out + col];
    }
}

__global__ void reduce_sum3d(double * dw_batch, double * dw, int batchSize, int N_in, int N_out){

    int index = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (index == 0 && row < N_in && col < N_out)
    {
        for (int b = 0; b < batchSize; ++b)
        {
            dw[row*N_out + col] += dw_batch[b*N_in*N_out + row*N_out + col];
        }
        // printf("%d\t%d\t%lf\n", row, col, dw[row*N_out + col]);  
    }  
}

__global__ void batch_multiply(double* dw_batch, double* w, double* delta, int batchSize, int N_in, int N_out){
    
    int index = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < batchSize && row < N_in && col ==0)
    {
        for (int j = 0; j < N_out; ++j)
        {
            delta[index*N_in + row] += dw_batch[index*N_in*N_out + row*N_out + j]*w[row*N_out + j];
        }  
    }  

}

__global__ void create_fired_before(int * firing_t_in, int * firing_t_out, int batchSize, bool * fired_before, int N_in, int N_out){

    int index = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    fired_before[index*N_in*N_out + row*N_out + col] = firing_t_in[index*N_in + row] < firing_t_out[index*N_out + col];    
}

//backprop_dense(firing_t1, firing_t0, delta2, delta1, w2,  lr2);
void backprop_dense(int * firing_t_out, int * firing_t_in, double * delta2,double * delta1, double * w2, double * dw2, double lr, bool calc_delta, int batchSize, int N_in, int N_out){

    double * norm, * dw2_batch;
    bool * fired_before;
    cudaMalloc((void **) &norm, sizeof(double)*batchSize);
    cudaMalloc((void **) &dw2_batch, sizeof(double)*batchSize*N_in*N_out);
    cudaMalloc((void **) &fired_before, sizeof(bool)*batchSize*N_in*N_out);

    dim3 threadsPerBlock_2D(10, 100);
    dim3 blocksPerGrid_2D(1, 1);

    blocksPerGrid_2D.x = ceil(double(N_out)/double(threadsPerBlock_2D.x));
    blocksPerGrid_2D.y = ceil(double(batchSize)/double(threadsPerBlock_2D.y));

    compute_norm<<<blocksPerGrid_2D, threadsPerBlock_2D>>>(delta2, norm, batchSize, N_out);
    grad_norm<<<blocksPerGrid_2D, threadsPerBlock_2D>>>(delta2, norm, batchSize, N_out);

    dim3 threadsPerBlock_3D(8, 16, 8);
    dim3 blocksPerGrid_3D(1, 1, 1);

    blocksPerGrid_3D.x = ceil(double(N_out)/double(threadsPerBlock_3D.x));
    blocksPerGrid_3D.y = ceil(double(N_in)/double(threadsPerBlock_3D.y));
    blocksPerGrid_3D.z = ceil(double(batchSize)/double(threadsPerBlock_3D.z));

    create_fired_before<<<blocksPerGrid_3D, threadsPerBlock_3D>>>(firing_t_in, firing_t_out, batchSize, fired_before, N_in, N_out);
    batch_bcast_dot<<<blocksPerGrid_3D, threadsPerBlock_3D>>>(delta2, fired_before, dw2_batch, batchSize, N_in, N_out);
    reduce_sum3d<<<blocksPerGrid_3D, threadsPerBlock_3D>>>(dw2_batch, dw2, batchSize, N_in, N_out);

    if(calc_delta) batch_multiply<<<blocksPerGrid_3D, threadsPerBlock_3D>>>(dw2_batch, w2, delta1, batchSize, N_in, N_out);

}

__global__ void update_weights(double * w, double * dw, double lr, int N_in, int N_out){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    w[row*N_out + col] += lr*dw[row*N_out + col];
}