#define n_train 60000
#define n_test  10000
#define tmax 256
#define Imax 255
//int n_layers = 2;
const int img_size   = 28*28;
const int n_inp      = 28*28;
const int n_hid      = 400;
const int n_out      = 10;
const int n_epochs   = 1;
const int batch_size = 100;
const int n_batches  = n_train/batch_size;
const int th_val     = 100;
const int gamma_     = 3;
const int lr1        = 0.01;
const int lr2        = 0.01;
const int a1 = 0, b1 = 5, a2 = 0, b2 = 50;

#include "mnist_load.cpp"
#include "encoding.cpp"
#include "initialization.cpp"
#include "forward_pass.cpp"
#include "backward_pass.cpp"

#include <fstream>
#include <iostream>
#include <sstream>
//typedef long double LD;

int model_eval(const vector<vector<int>>& firing_t, const vector<int>& y_batch, int batch_size, int n_out){

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
    }
    return correct;
}

int main(){

	FILE *myFile;
	//float w1_[400*10];
	//float  []
	vector<float> w1_(400*10);
	vector<float> w0_(784*400);

	int n=0,i = 0;

	myFile = fopen("w1.txt", "r");
	if (myFile == NULL) {
	printf("failed to open file\n");
	return 1;
	}

	while (fscanf(myFile, "%f", &w1_[n++]) != EOF);
	fclose(myFile);
	
	for (int i = 0; i < 400*10; ++i)
	{
		//cout<<w1_[i]<<"\n";
	}
	

	n=0,i = 0;

	myFile = fopen("w0.txt", "r");
	if (myFile == NULL) {
	printf("failed to open file\n");
	return 1;
	}

	while (fscanf(myFile, "%f", &w0_[n++]) != EOF);
	fclose(myFile);
	/*
	for (int i = 0; i < 784*400; ++i)
	{
		cout<<w0_[i]<<"\n";
	}
	*/
	vector<vector<double>> w1(img_size, vector<double>(n_hid));//, dw1(img_size, vector<double>(n_hid,0)); // w1 = (784,1000)
	vector<vector<double>> w2(n_hid, vector<double>(n_out));// dw2(n_hid, vector<double>(n_out,0));       // w2 = (1000,10)
	
	for (int i = 0; i < img_size; ++i)
	{
		for (int j = 0; j < n_hid; ++j)
		{
			w1[i][j] = w0_[i*n_hid + j];
			//cout<<w1[i][j]<<"\n";
		}
	}

	for (int i = 0; i < n_hid; ++i)
	{
		for (int j = 0; j < n_out; ++j)
		{
			w2[i][j] = w1_[i*n_out + j];
			//cout<<w2[i][j]<<"\n";
		}
	}

	
	vector<vector<double>> x_train,  x_test;
	vector<int> y_train, y_test;
	//cout<<std::filesystem::exists("train-images-idx3-ubyte");
	ReadMNIST(60000,784,x_train,"train-images-idx3-ubyte");
	ReadMNIST(10000,784,x_test,"t10k-images-idx3-ubyte");
	ReadMNIST_label(60000,  y_train,"train-labels-idx1-ubyte");
	ReadMNIST_label(10000,  y_test,"t10k-labels-idx1-ubyte");

	//preproc(x_train, x_test);
	int test_acc = 0;
	for (int b = 0; b < n_batches; ++b)
	{
		vector<vector<vector<int>>> x_batch, x1_out, x2_out;
		vector<vector<int>> firing_t0, firing_t1, firing_t2;
		vector<int> y_batch; // (batch_size, n_out)
		batch_spike_encoding(x_train, y_train, x_batch, y_batch, firing_t0, b, batch_size, n_inp);
		dense(x_batch, x1_out, w1, firing_t1, batch_size);
		dense(x1_out , x2_out, w2, firing_t2, batch_size);
		test_acc += model_eval(firing_t2, y_batch, batch_size, n_out);
		cout<<"\tBatch:"<<b<<":"<<test_acc<<"/"<<((b+1)*batch_size)<<" = "<<(test_acc/(1.0*(b+1)*batch_size))<<" so far\n";
	}
	cout<<"Train Accuracy = "<<test_acc<<"/60000 = "<<(test_acc/60000.0)<<"\n";
	

	/*
	for (int b = 0; b < n_test_batches; ++b)
	{
		vector<vector<vector<int>>> x_batch, x1_out, x2_out;
		vector<vector<int>> firing_t0, firing_t1, firing_t2;
		vector<int> y_batch; // (batch_size, n_out)
		batch_spike_encoding(x_train, y_train, x_batch, y_batch, firing_t0, b, batch_size, n_inp);
		dense(x_batch, x1_out, w1, firing_t1, batch_size);
		dense(x1_out , x2_out, w2, firing_t2, batch_size);
		test_acc += loss_calc(firing_t2, y_batch, delta2, batch_size, n_out);
	}
	cout<<"\t\tTest Accuracy = "<<test_acc<<"/60000 = "<<(train_acc/60000.0)<<"\n";
	*/



	
}
