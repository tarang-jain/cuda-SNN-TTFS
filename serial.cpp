#define n_train 60000
#define n_test  10000
#define tmax 256
#define Imax 255
//int n_layers = 2;
const int img_size   = 28*28;
const int n_inp      = 28*28;
const int n_hid      = 1000;
const int n_out      = 10;
const int n_epochs   = 1;
const int batch_size = 100;
const int n_batches  = n_train/batch_size;
const int th_val     = 100;
const int a1 = 0, b1 = 5, a2 = 0, b2 = 50;

#include "mnist_load.cpp"
#include "encoding.cpp"
#include "initialization.cpp"
#include "forward_pass.cpp"

//typedef long double LD;

int main(){

	vector<vector<double>> w1(img_size, vector<double>(n_hid)); // w1 = (784,1000)
	vector<vector<double>> w2(n_hid, vector<double>(n_out)); // w2 = (1000,10)
	
	weight_init(w1, w2);

	vector<vector<double>> x_train,  x_test;
	vector<int> y_train, y_test;
	//cout<<std::filesystem::exists("train-images-idx3-ubyte");
	ReadMNIST(60000,784,x_train,"train-images-idx3-ubyte");
	ReadMNIST(10000,784,x_test,"t10k-images-idx3-ubyte");
	ReadMNIST_label(60000,  y_train,"train-labels-idx1-ubyte");
	ReadMNIST_label(10000,  y_test,"t10k-labels-idx1-ubyte");

	//preproc(x_train, x_test);

	cout <<"Training the model....\n";
	//Training:
	for (int e = 0; e < n_epochs; ++e)
	{
		cout <<"epoch: "<<e+1<<" \n";
		rand_shuffle(x_train, y_train);

		for (int b = 0; b < n_batches; ++b)
		{
			cout<<"batch: "<<b+1<<"\n";
			/*
			vector<vector<vector<double>>> x_batch(batch_size, vector<vector<double>>(n_inp, vector<double>(tmax))), 
										   x1_out(batch_size, vector<vector<double>>(n_hid, vector<double>(tmax))), 
										   x2_out(batch_size, vector<vector<double>>(n_inp, vector<double>(tmax))); // (batch_size,img_size,tmax)
			*/
			vector<vector<vector<int>>> x_batch, x1_out, x2_out;
			vector<vector<int>> firing_t1, firing_t2;
			vector<int> y_batch; // (batch_size, n_out)
			
			batch_spike_encoding(x_train, y_train, x_batch, y_batch, b, batch_size, n_inp);
			//if(b==0) cout<<x_batch.size()<<" "<<x_batch[0].size()<<" "<<x_batch[0][0].size()<<"\n";

			dense(x_batch, x1_out, w1, firing_t1, batch_size);
			dense(x1_out , x2_out, w2, firing_t2, batch_size);

			//loss_calc(x2_out, y_batch, grad2);

			//backprop_dense(grad2, w2, grad1, lr2);
			//backprop_dense(grad1, w1, grad0, lr1);

			/*
			//backprop(grad2, w2, grad1, lr2, dw2);
			//update_weights(w2, dw1, lr2);
			*/

		}
	}




	
}
