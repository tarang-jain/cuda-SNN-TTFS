void weight_init(vector<vector<double>>& w1, vector<vector<double>>& w2){
    random_device seed;
    mt19937 generator(seed());
    uniform_real_distribution<double> uni(0,1);
    for (int i = 0; i < w1.size(); ++i)
    {
        for (int j = 0; j < w1[0].size(); ++j)
        {
            w1[i][j] = (b1-a1)*uni(generator);
        }
    }
    for (int i = 0; i < w2.size(); ++i)
    {
        for (int j = 0; j < w2[0].size(); ++j)
        {
            w2[i][j] = (b2-a2)*uni(generator);
        }
    }
}
