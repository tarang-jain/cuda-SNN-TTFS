#include <bits/stdc++.h>
#include <stdio.h>
#include <math.h> 
#include <random>
using namespace std;

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr, string fullpath)
{
    arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    //ifstream file ("C:\\t10k-images.idx3-ubyte",ios::binary);
    //ifstream file ("E:\\College\\ME766_High_Performance_Scientific_Computing\\Project\\mnist\\train-images-idx3-ubyte",ios::binary);
    ifstream file (fullpath,ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
    file.close();
}

int ReverseInt_label (int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void ReadMNIST_label(int NumberOfImages,vector<int> &arr, string fullpath)
{
    arr.resize(NumberOfImages);
    ifstream file (fullpath,ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt_label(magic_number);
        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt_label(number_of_images);

        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,1);
            arr[i]= (int)temp;
        }
    }
    file.close();
}


/*
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
    
    cout<<x_train.size()<<" "<<x_train[0].size()<<"\n";
    cout<<y_train.size()<<" "<<"\n";
    cout<<x_test.size()<<" "<<x_test[0].size()<<"\n";
    cout<<y_test.size()<<"\n";
    
    double count = 0;
    for (int i = 0; i < x_train.size(); ++i)
    {
        for (int j = 0; j < x_train[0].size(); ++j)
        {
            count = max(x_train[i][j], count);
        }
    }
    cout<<count<<"\n";

    int c = 0;
    for (int i = 0; i < y_test.size(); ++i)
    {
        c=max(y_test[i],c);
        //cout<<y_test[i]<<"\n";
    }
    cout<<c<<"\n";
}
*/