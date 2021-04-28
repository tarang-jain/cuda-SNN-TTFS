#include "mnist_load.cpp"

#include <random> 

#define n_train 60000
#define n_test  10000
#define tmax 100

//int n_layers = 2;
const int img_size = 28*28;
const int n_inp    = 28*28;
const int n_hid    = 1000;
const int n_out    = 10;
const int n_epochs = 1;
const int batch_size = 100;
const int n_batches = n_train/batch_size;

//typedef long double LD;

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

void batch_spike_encoding(const vector<vector<double>>& x_train, const vector<int>& y_train, 
					      vector<vector<vector<double>>>& x_batch, vector<vector<double>>& y_batch, int b){
	//Time-to-first Spike Coding
	
}

int main(){
	
	//cout<<std::filesystem::exists("train-images-idx3-ubyte");

	vector<vector<double>> w1(img_size, vector<double>(n_hid)); // w1 = (784,1000)
	vector<vector<double>> w2(n_hid, vector<double>(n_out)); // w2 = (1000,10)

	vector<vector<double>> x_train,  x_test;
	vector<int> y_train, y_test;
	ReadMNIST(60000,784,x_train,"train-images-idx3-ubyte");
	ReadMNIST(10000,784,x_test,"t10k-images-idx3-ubyte");
	ReadMNIST_label(60000,  y_train,"train-labels-idx1-ubyte");
	ReadMNIST_label(10000,  y_test,"t10k-labels-idx1-ubyte");

	preproc(x_train, x_test);

	cout <<"Training the model....\n";
	//Training:
	for (int e = 0; e < n_epochs; ++e)
	{
		cout <<"epoch: "<<e+1<<" \n";
		rand_shuffle(x_train, y_train);

		for (int b = 0; b < n_batches; ++b)
		{
			vector<vector<vector<double>>> x_batch, x1_out, x2_out; // (batch_size,img_size,tmax)
			vector<vector<double>> y_batch; // (batch_size, n_out)
			
			//batch_spike_encoding(x_train, y_train, x_batch, y_batch, b)

			//dense(x_batch, x1_out, w1);
			//dense(x1_out , x2_out, w2);

			//loss_calc(x2_out, y_batch, grad2);

			//backprop(grad2, w2, grad1, lr2);
			//backprop(grad1, w1, grad0, lr1);

			/*
			//backprop(grad2, w2, grad1, lr2, dw2);
			//update_weights(w2, dw1, lr2);
			*/

		}
	}




	
}
