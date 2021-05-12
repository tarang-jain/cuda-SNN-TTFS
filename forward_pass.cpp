/*
void batch_tensordot(const vector<vector<vector<int>>>& x_in, const vector<vector<double>>& w, int batch_size){
}
void batch_cumsum(int batch_size){
}
void batch_thresholding(int batch_size){
} 
*/

void tensordot(const vector<vector<int>>& x, const vector<vector<double>>& w, vector<vector<double>>& V){
    int Nin  = w.size();
    int Nout = w[0].size();
    for (int t = 0; t < tmax; ++t)
    {
        for (int i = 0; i < Nout; ++i)
        {
            double val = 0;
            for (int j = 0; j < Nin; ++j)
            {
                val += x[j][t]*w[j][i];
            }
            V[i][t] = val;
        }
    }

}

void cumsum(vector<vector<double>>& V){
    //Can be of lower complexity. Divide and conquer.
    int N = V.size();
    for (int i = 0; i < N; ++i)
    {
        for (int t = 1; t < tmax; ++t)
        {
            V[i][t] += V[i][t-1];
        }
    }

}

void thresholding(const vector<vector<double>>& V, vector<vector<int>>& x, vector<int>& firing_t){
    int N = V.size();
    for (int i = 0; i < N; ++i)
    {
        for (int t = 0; t < tmax; ++t)
        {
            if (V[i][t]>=th_val)
            {
                x[i][t] = 1;
                firing_t[i] = t;
                break;
            }
        }
    }

}

void dense(const vector<vector<vector<int>>>& x_in, vector<vector<vector<int>>>& x_out, const vector<vector<double>>& w,
           vector<vector<int>>& firing_t, int batch_size){
    int Nin  = w.size();
    int Nout = w[0].size();
    x_out = vector<vector<vector<int>>>(batch_size, vector<vector<int>>(Nout, vector<int>(tmax, 0)));
    firing_t = vector<vector<int>>(batch_size, vector<int>(Nout, tmax)); //tmax : means no spike


    for (int b = 0; b < batch_size; ++b)
    {
        vector<vector<double>> volt(Nout, vector<double>(tmax, 0));
        tensordot(x_in[b], w, volt);
        cumsum(volt);
        thresholding(volt, x_out[b], firing_t[b]);
    }
    //batch_tensordot();
    //batch_cumsum();
    //batch_thresholding();
}

