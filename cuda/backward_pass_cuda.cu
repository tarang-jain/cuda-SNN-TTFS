int loss_calc(int * firing_t, int * y_batch,float * delta, int batchSize, int N, int t_max, int * correct){

    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int min_time[batchSize];
    vector<vector<int>> target = firing_t;
    *correct = 0;

    if (index < batchSize && row < N)
    {
        if (row == 0){
            int winner = 0;
            min_time[index] = t_max;
            for (int i = 0; i < N; i++)
            {
                int time = firing_t[index*N + i];
                winner = time < min_time[index] ? i : winner;
                min_time[index] = time < min_time[index] ? time : min_time[index];
            }
            if(winner==y_batch[index]) correct++;
        }
        __syncthreads();

        if(min_time[index]==t_max)
        {
            target[index*N + row] = min_time[index];
            if (row == y_batch[index])
            {
                target[index*N + row] = min_time[index] - gamma_;
            }
        }
        else
        {
            target[index*N + row] = firing_t[index*N + row];
            target[index*N + row] = (firing_t[b][i] - min_time) < gamma_ ? (min(min_time + gamma_, tmax)) : firing_t[index*N + row];

            if (row == y_batch[index])
            {
                target[index*N + row] = min_time[index];
            }
        }

        delta[index * N + row] = target[index * N + row] - firing_t[index * N + row]; //delta = target - firing_t;
    }
}

void compute_norm(float * delta, float * norm, int batchSize, int N){

    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batchSize && row < N && threadIdx.x == 0){
        float sum = 0;
        for(int i = 0; i < blockDim.x; i++) sum += pow(delta[index*N + i], 2);
        atomicAdd(norm[index], sum);
    }
}

void grad_norm(float * delta, float * norm, int batchSize, int N){

    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < batchSize && row < N){
        delta[index*N + row] /= norm[index];
    }
}

void batch_bcast_dot(float * delta, bool * fired_before, float * dw_batch){

    int index = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int b_size = fired_before.size();
    int Nin    = fired_before[0].size();
    int Nout   = fired_before[0][0].size();

    for (int b = 0; b < b_size; ++b)
    {
        for (int i = 0; i < Nin; ++i)
        {
            for (int j = 0; j < Nout; ++j)
            {
                dw_batch[b][i][j] = delta[b][j]*fired_before[b][i][j];
            }
        }
    }
}

void reduce_sum3d(const vector<vector<vector<double>>>& dw_batch, vector<vector<double>>& dw){
    //reduce_sum along axis 0
    int b_size = dw_batch.size();
    int Nin    = dw_batch[0].size();
    int Nout   = dw_batch[0][0].size();

    dw = vector<vector<double>>(Nin, vector<double>(Nout, 0));

    for (int b = 0; b < b_size; ++b)
    {
        for (int i = 0; i < Nin; ++i)
        {
            for (int j = 0; j < Nout; ++j)
            {
                dw[i][j] += dw_batch[b][i][j];
            }
        }
    }
}

void batch_multiply(const vector<vector<vector<double>>>& dw_batch, const vector<vector<double>>& w, vector<vector<double>>& delta){
    int b_size = dw_batch.size();
    int Nin = w.size();
    int Nout = w[0].size();

    //vector<vector<vector<double>>> delta_batch(b_size, vector<vector<bool>>(Nin, vector<bool>(Nout, false)));
    delta = vector<vector<double>>(b_size, vector<double>(Nin, 0));

    for (int b = 0; b < b_size; ++b)
    {
        for (int i = 0; i < Nin; ++i)
        {
            for (int j = 0; j < Nout; ++j)
            {
                //delta_batch[b][i][j] = dw_batch[b][i][j]*dw[i][j];
                delta[b][i] += dw_batch[b][i][j]*w[i][j];
            }
        }
    }
    

}

//backprop_dense(firing_t1, firing_t0, delta2, delta1, w2,  lr2);
void backprop_dense(const vector<vector<int>>& firing_t1, const vector<vector<int>>& firing_t0, vector<vector<double>>& delta2,
                    vector<vector<double>>& delta1, vector<vector<double>>& w2, vector<vector<double>>& dw2, int lr, bool calc_delta){

    int b_size = firing_t0.size();
    int Nin = w2.size();
    int Nout = w2[0].size();
    grad_norm(delta2);

    vector<vector<vector<bool>>> fired_before(b_size, vector<vector<bool>>(Nin, vector<bool>(Nout, false)));
    
    for (int b = 0; b < b_size; ++b)
    {
        for (int i = 0; i < Nin; ++i)
        {
            for (int j = 0; j < Nout; ++j)
            {
                fired_before[b][i][j] = firing_t0[b][i] < firing_t1[b][j];
            }
        }
    }
    vector<vector<vector<double>>> dw2_batch(b_size, vector<vector<double>>(Nin, vector<double>(Nout, 0)));
    //vector<vector<double>> dw2; //(Nin, vector<double>(Nout, 0));
    batch_bcast_dot(delta2, fired_before, dw2_batch);
    reduce_sum3d(dw2_batch, dw2);


    if(calc_delta) batch_multiply(dw2_batch, w2, delta1); //Check if this has to be dw2_batch then reduction

}

void update_weights(vector<vector<double>>& w, const vector<vector<double>>& dw, double lr){
    int Nin = w.size();
    int Nout = w[0].size();

    for (int i = 0; i < Nin; ++i)
    {
        for (int j = 0; j < Nout; ++j)
        {
            w[i][j] += lr*dw[i][j];
        }
    }
}
