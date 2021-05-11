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

#include <stdio.h>
#include <stdlib.h>
#include "mnist_load.cpp"
#include "encoding.cpp"
#include "cuda_runtime.h"
#include "initialization.cpp"
#include "forward_pass_cuda.cu"

int main(int argc, char** argv){

	float *w1, *w2, *dev_w1, *dev_w2;

	w1 = (float*) malloc(sizeof(float)*img_size*n_hid);
    w2 = (float*) malloc(sizeof(float)*n_hid*n_out);

	cudaMalloc((void **) &dev_w1, sizeof(float)*img_size*n_hid);
    cudaMalloc((void **) &dev_w2, sizeof(float)*n_hid*n_out);
	
	weight_init_1D(w1, w2, img_size*n_hid, n_hid*n_out);

	cudaMemcpy(dev_w1, w1, sizeof(float)*n_inp*n_hid, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_w2, w2, sizeof(float)*n_hid*n_out, cudaMemcpyHostToDevice);

	vector<vector<double>> x_train,  x_test;
	vector<int> y_train, y_test;

	ReadMNIST(60000,784,x_train,"../train-images-idx3-ubyte");
	ReadMNIST(10000,784,x_test,"../t10k-images-idx3-ubyte");
	ReadMNIST_label(60000,  y_train,"../train-labels-idx1-ubyte");
	ReadMNIST_label(10000,  y_test,"../t10k-labels-idx1-ubyte");

	cout <<"Training the model....\n";
	//Training:
	for (int e = 0; e < n_epochs; ++e)
	{
		cout <<"epoch: "<<e+1<<" \n";
		rand_shuffle(x_train, y_train);

		for (int b = 0; b < n_batches; ++b)
		{
			cout<<"batch: "<<b+1<<"\n";

			int *x_batch, *dev_x_batch, *dev_x1_out, *dev_x2_out, *firing_t0, *dev_firing_t0, *dev_firing_t1, *dev_firing_t2;
			x_batch = (int *) calloc(batch_size*n_inp*tmax, sizeof(int));
			firing_t0 = (int *) malloc(sizeof(int)*batch_size*n_inp);
			vector<int> y_batch; // (batch_size, n_out)
			
			cudaMalloc((void **) &dev_x_batch, sizeof(int)*batch_size*n_inp*tmax);
			cudaMalloc((void **) &dev_x1_out, sizeof(int)*batch_size*n_hid*tmax);
			cudaMalloc((void **) &dev_x2_out, sizeof(int)*batch_size*n_out*tmax);
			cudaMalloc((void **) &dev_firing_t0, sizeof(int)*batch_size*n_inp);
			cudaMalloc((void **) &dev_firing_t1, sizeof(int)*batch_size*n_hid);
			cudaMalloc((void **) &dev_firing_t2, sizeof(int)*batch_size*n_out);
			cudaMemset(dev_x1_out, 0, sizeof(int)*batch_size*n_hid*tmax);
			cudaMemset(dev_x1_out, 0, sizeof(int)*batch_size*n_out*tmax);

			batch_spike_encoding_1D(x_train, y_train, x_batch, y_batch, firing_t0, b, batch_size, n_inp);

			cudaMemcpy(dev_x_batch, x_batch, sizeof(int)*batch_size*n_inp*tmax, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_firing_t0, firing_t0, sizeof(int)*batch_size*n_inp, cudaMemcpyHostToDevice);

			batch_dense(dev_x_batch, dev_x1_out, dev_firing_t1, dev_w1, n_inp, n_hid, batch_size, th_val, tmax);
			batch_dense(dev_x1_out, dev_x2_out, dev_firing_t2, dev_w2, n_hid, n_out, batch_size, th_val, tmax);
		

	// 		//loss_calc(x2_out, y_batch, grad2);

	// 		//backprop_dense(grad2, w2, grad1, lr2);
	// 		//backprop_dense(grad1, w1, grad0, lr1);

	// 		/*
	// 		//backprop(grad2, w2, grad1, lr2, dw2);
	// 		//update_weights(w2, dw1, lr2);
	// 		*/

	        free(x_batch);
			free(firing_t0);
			cudaFree(dev_x_batch);
	        cudaFree(dev_x1_out);
	        cudaFree(dev_x2_out);
			cudaFree(dev_firing_t0);
			cudaFree(dev_firing_t1);
			cudaFree(dev_firing_t2);
		}
	}

    free(w1);
	free(w2);
	cudaFree(dev_w1);
	cudaFree(dev_w2);
	return 0;
	
}


