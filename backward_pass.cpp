
void batch_sub(const vector<vector<int>>& target, const vector<vector<int>>& firing_t, vector<vector<double>>& delta){
    int b_size = firing_t.size();
    int N = firing_t[0].size();

    delta = vector<vector<double>>(b_size, vector<double>(N,0));

    for (int b = 0; b < b_size; ++b)
    {
        for (int i = 0; i < N; ++i)
        {
            delta[b][i] = target[b][i] - firing_t[b][i];
        }
    }
}

int loss_calc(const vector<vector<int>>& firing_t, const vector<int>& y_batch,vector<vector<double>>& delta, int batch_size, int n_out){

    vector<vector<int>> target = firing_t;
    int correct = 0;

    //cout<<"okay0\n";
    for (int b = 0; b < batch_size; ++b)
    {
        int winner = 0; int min_time = tmax;
        for (int i = 0; i < n_out; ++i)
        {
            if(firing_t[b][i]<min_time)
            {
                winner = i;
                min_time = firing_t[b][i];
            }
        }
        if(winner==y_batch[b]) correct++;
        //cout<<b<<" okay1\n";
        if(min_time==tmax)
        {
            for(int i=0; i<n_out;++i) target[b][i] = min_time;
            target[b][y_batch[b]] = min_time - gamma_;
        }
        else
        {
            target[b] = firing_t[b];
            for (int i = 0; i < n_out; ++i)
            {
                if(firing_t[b][i] - min_time < gamma_)
                {
                    target[b][i] = min(min_time+gamma_, tmax);
                }
            }
            target[b][y_batch[b]] = min_time;
        }
        //cout<<b<<" okay2\n";
    }
    batch_sub(target, firing_t, delta); //delta = target - firing_t;
    return correct;
}

void grad_norm(vector<vector<double>>& delta){
    int b_size = delta.size();
    int N = delta[0].size();

    for (int b = 0; b < b_size; ++b)
    {
        double norm = 0;
        for(int i=0; i<N; ++i) norm += delta[b][i]*delta[b][i];
        norm = sqrt(norm);
        for(int i=0; i<N; ++i) delta[b][i] /= norm;
    }
}

void batch_bcast_dot(const vector<vector<double>>& delta,const vector<vector<vector<bool>>>& fired_before,
                     vector<vector<vector<double>>>& dw_batch){

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
