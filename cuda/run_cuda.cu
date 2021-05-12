#define n_train 60000
#define n_test  10000
#define tmax 256
#define Imax 255
//int n_layers = 2;
const int img_size   = 28*28;
const int n_inp      = 28*28;
const int n_hid      = 400;
const int n_out      = 10;
const int n_epochs   = 30;
const int batch_size = 100;
const int n_batches  = n_train / batch_size;
const int th_val     = 100;
const int gamma_     = 3;
const double lr1        = 0.2;
const double lr2        = 0.2;
const int a1 = 0, b1 = 5, a2 = 0, b2 = 50;

#include <stdio.h>
#include <stdlib.h>
#include "mnist_load.cpp"
#include "encoding.cpp"
#include "cuda_runtime.h"
#include "initialization.cpp"
#include "forward_pass_cuda.cu"
#include "backward_pass_cuda.cu"
int main(int argc, char** argv){

	int *x_batch, * dev_x_batch, *dev_x1_out, *dev_x2_out, *firing_t0, *dev_firing_t0, *dev_firing_t1, *dev_firing_t2, *y_batch, *dev_y_batch;
	double *w1, *w2, *dev_w1, *dev_w2, *dev_dw1, *dev_dw2, *dev_delta0, *dev_delta1, *dev_delta2, *dev_correct, *dev_lr1, *dev_lr2;

	w1 = (double*) malloc(sizeof(double)*img_size*n_hid);
    w2 = (double*) malloc(sizeof(double)*n_hid*n_out);
	x_batch = (int *) malloc(sizeof(int)*batch_size*n_inp*tmax);
	firing_t0 = (int *) malloc(sizeof(int)*batch_size*n_inp);
	y_batch = (int *) malloc(sizeof(int)*batch_size);
	cudaMalloc((void **) &dev_w1, sizeof(double)*img_size*n_hid);
    cudaMalloc((void **) &dev_w2, sizeof(double)*n_hid*n_out);
	cudaMalloc((void **) &dev_dw1, sizeof(double)*img_size*n_hid);
    cudaMalloc((void **) &dev_dw2, sizeof(double)*n_hid*n_out);
	cudaMalloc((void **) &dev_x_batch, sizeof(int)*batch_size*n_inp*tmax);
	cudaMalloc((void **) &dev_x1_out, sizeof(int)*batch_size*n_hid*tmax);
	cudaMalloc((void **) &dev_x2_out, sizeof(int)*batch_size*n_out*tmax);
	cudaMalloc((void **) &dev_firing_t0, sizeof(int)*batch_size*n_inp);
	cudaMalloc((void **) &dev_firing_t1, sizeof(int)*batch_size*n_hid);
	cudaMalloc((void **) &dev_firing_t2, sizeof(int)*batch_size*n_out);
	cudaMalloc((void **) &dev_delta0, sizeof(double)*batch_size*n_inp);
	cudaMalloc((void **) &dev_delta1, sizeof(double)*batch_size*n_hid);
	cudaMalloc((void **) &dev_delta2, sizeof(double)*batch_size*n_out);
	cudaMalloc((void **) &dev_correct, sizeof(int));
	cudaMalloc((void **) &dev_lr1, sizeof(double));
	cudaMalloc((void **) &dev_lr2, sizeof(double));
	cudaMalloc((void **) &dev_y_batch, sizeof(int)*batch_size);

	weight_init_1D(w1, w2, img_size*n_hid, n_hid*n_out);

	cudaMemcpy(dev_w1, w1, sizeof(double)*n_inp*n_hid, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_w2, w2, sizeof(double)*n_hid*n_out, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lr1, &lr1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lr2, &lr2, sizeof(double), cudaMemcpyHostToDevice);

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
		// rand_shuffle(x_train, y_train);

		int training_acc = 0;
		int *dev_training_acc;

		cudaMalloc((void **) &dev_training_acc, sizeof(int));
		cudaMemcpy(dev_training_acc, &training_acc, sizeof(int), cudaMemcpyHostToDevice);

		for (int b = 0; b < n_batches; ++b)
		{
			cout<<"batch: "<<b+1<<"\n";




			memset(x_batch, 0, batch_size*n_inp*tmax);

			cudaMemset(dev_dw1, 0, sizeof(double)*img_size*n_hid);
			cudaMemset(dev_dw2, 0, sizeof(double)*n_hid*n_out);			
			cudaMemset(dev_x1_out, 0, sizeof(int)*batch_size*n_hid*tmax);
			cudaMemset(dev_x1_out, 0, sizeof(int)*batch_size*n_out*tmax);

			batch_spike_encoding_1D(x_train, y_train, x_batch, y_batch, firing_t0, b, batch_size, n_inp);

			cudaMemcpy(dev_x_batch, x_batch, sizeof(int)*batch_size*n_inp*tmax, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_y_batch, y_batch, sizeof(int)*batch_size, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_firing_t0, firing_t0, sizeof(int)*batch_size*n_inp, cudaMemcpyHostToDevice);

			batch_dense(dev_x_batch, dev_x1_out, dev_firing_t1, dev_w1, n_inp, n_hid, batch_size, th_val, tmax);
			batch_dense(dev_x1_out, dev_x2_out, dev_firing_t2, dev_w2, n_hid, n_out, batch_size, th_val, tmax);

			dim3 threadsPerBlock_2D(n_out, batch_size);
			dim3 blocksPerGrid_2D(1, 1);


			if (batch_size > floor(double(1024)/double(n_out))){
				threadsPerBlock_2D.y = floor(double(1024)/double(n_out));
				blocksPerGrid_2D.y = ceil(double(batch_size)/double(threadsPerBlock_2D.y));
			}


			loss_calc<<<blocksPerGrid_2D, threadsPerBlock_2D>>>(dev_firing_t2, dev_y_batch, dev_delta2, batch_size, n_out, tmax, dev_training_acc);

			cudaMemcpy(&training_acc, dev_training_acc, sizeof(int), cudaMemcpyDeviceToHost);

			cout<<"\t\ttrain acc = "<<training_acc<<"/"<<((b+1)*batch_size)<<" = "<<(training_acc/(1.0*(b+1)*batch_size))<<"\n";

			backprop_dense(dev_firing_t2, dev_firing_t1, dev_delta2, dev_delta1, dev_w2, dev_dw2, lr2, true, batch_size, n_hid, n_out);

			if (n_hid > floor(double(1024)/double(n_out))){
				threadsPerBlock_2D.y = floor(double(1024)/double(n_out));
				blocksPerGrid_2D.y = ceil(double(n_hid)/double(threadsPerBlock_2D.y));
			}

			update_weights<<<blocksPerGrid_2D, threadsPerBlock_2D>>>(dev_w2, dev_dw2, lr2, n_hid, n_out);

			backprop_dense(dev_firing_t1, dev_firing_t0, dev_delta1, dev_delta0, dev_w1, dev_dw1, lr1, false, batch_size, n_inp, n_hid);

			threadsPerBlock_2D.x = n_hid;
			threadsPerBlock_2D.y = n_inp;

			if (n_inp > floor(double(1024)/double(n_hid))){
				threadsPerBlock_2D.y = floor(double(1024)/double(n_hid));
				blocksPerGrid_2D.y = ceil(double(n_inp)/double(threadsPerBlock_2D.y));
			}

			update_weights<<<blocksPerGrid_2D, threadsPerBlock_2D>>>(dev_w1, dev_dw1, lr1, n_inp, n_hid);

		}
	}

	free(x_batch);
	free(firing_t0);
	free(w1);
	free(w2);
	cudaFree(dev_x_batch);
	cudaFree(dev_x1_out);
	cudaFree(dev_x2_out);
	cudaFree(dev_firing_t0);
	cudaFree(dev_firing_t1);
	cudaFree(dev_firing_t2);
	cudaFree(dev_w1);
	cudaFree(dev_w2);
	cudaFree(dev_dw1);
	cudaFree(dev_dw2);
	cudaFree(dev_dw2);
	cudaFree(dev_delta0);
	cudaFree(dev_delta1);
	cudaFree(dev_delta2);
	cudaFree(dev_correct);
	cudaFree(dev_lr1);
	cudaFree(dev_lr2);

	return 0;
	
}


