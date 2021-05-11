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
                          int * x_batch, vector<int>& y_batch,
                          int b, int batch_size, int n_inp){
    
    // cout<<x_train[0].size()<<tmax;
    for (int i = b*batch_size, k = 0; i < (b+1)*batch_size; ++i, ++k)
    {
        for (int j = 0; j < x_train[0].size(); ++j)
        {
            int t_spike = (Imax+1) - x_train[i][j];
            x_batch[k*x_train[0].size()*tmax + j*tmax + t_spike] = 1;
        }
    }
    for(int i=b*batch_size; i< (b+1)*batch_size; ++i) y_batch.push_back(y_train[i]);
}
