#include <random>

void preproc(vector<vector<double>>& x_train, vector<vector<double>>& x_test){
    double x_max = x_train[0][0];
    double x_min = x_train[0][0];
    for (int i = 0; i < x_train.size(); ++i)
    {
        for (int j = 0; j < x_train[0].size(); ++j)
        {
            x_max = max(x_max, x_train[i][j]);
            x_min = max(x_min, x_train[i][j]);
        }
    }
    double x_delta = x_max - x_min;

    for (int i = 0; i < x_train.size(); ++i)
    {
        for (int j = 0; j < x_train[0].size(); ++j)
        {
            x_train[i][j] = (x_train[i][j] - x_min)/x_delta;
        }
    }

    for (int i = 0; i < x_test.size(); ++i)
    {
        for (int j = 0; j < x_test[0].size(); ++j)
        {
            x_test[i][j] = (x_test[i][j] - x_min)/x_delta;
        }
    }
}

void rand_shuffle(vector<vector<double>>& x_train, vector<int>& y_train){
    vector<int> perm;
    for(int i=0; i<x_train.size();++i) perm.push_back(i);

    // obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(perm.begin(), perm.end(), std::default_random_engine(seed));

    vector<vector<double>> x_temp = x_train;
    vector<int>& y_temp = y_train;

    for (int i = 0; i < x_train.size(); ++i)
    {
        x_train[i] = x_temp[perm[i]];
        y_train[i] = y_temp[perm[i]];
    }

}
//Time-to-first Spike Coding : TTFS coding
void batch_spike_encoding(const vector<vector<double>>& x_train, const vector<int>& y_train, 
                          vector<vector<vector<int>>>& x_batch, vector<int>& y_batch, vector<vector<int>>& firing_t0,
                          int b, int batch_size, int n_inp){

    //vector<vector<double>> x_used = x_train(x_train.begin() + (b*batch_size), x_train.begin() + ((b+1)*batch_size) + 1);
    //vector<vector<double>> x_used;
    //for(int i=b*batch_size; i< (b+1)*batch_size; ++i) x_used.push_back(x_train[i]);

    
    x_batch = vector<vector<vector<int>>>(batch_size, vector<vector<int>>(n_inp, vector<int>(tmax, 0)));
    firing_t0 = vector<vector<int>>(batch_size, vector<int>(n_inp, 0));
    
    for (int i = b*batch_size, k = 0; i < (b+1)*batch_size; ++i, ++k)
    {
        for (int j = 0; j < x_train[0].size(); ++j)
        {
            int t_spike = (Imax+1) - x_train[i][j];
            firing_t0[k][j] = t_spike;
            x_batch[k][j][t_spike] = 1;
            //cout<<t_spike<<"\n";
        }
    }
    

    //y_batch = y_train(y_train.begin() + (b*batch_size), y_train.begin() + ((b+1)*batch_size) + 1);
    for(int i=b*batch_size; i< (b+1)*batch_size; ++i) y_batch.push_back(y_train[i]);
    //if(b==0) cout<<y_batch.size()<<"\n";
}

void batch_spike_encoding_1D(const vector<vector<double>>& x_train, const vector<int>& y_train, int * x_batch, vector<int>& y_batch, int * firing_t, int b, int batch_size, int n_inp){
    
    // cout<<x_train[0].size()<<tmax;
    for (int i = b*batch_size, k = 0; i < (b+1)*batch_size; ++i, ++k)
    {
        for (int j = 0; j < x_train[0].size(); ++j)
        {
            int t_spike = (Imax+1) - x_train[i][j];
            firing_t[k*x_train[0].size() + j] = t_spike;
            x_batch[k*x_train[0].size()*tmax + j*tmax + t_spike] = 1;
        }
    }
    for(int i=b*batch_size; i< (b+1)*batch_size; ++i) y_batch.push_back(y_train[i]);
}
