#include <iostream>
#include <fstream>

#include "include/utils.h"
#include "include/layers.h"
#include "include/lenet.h"


/**
 * Read weights from file
 *
 * @return: array with all weights read
 *
*/
LENET_T* read_weights() {
	//Variable for checking
	size_t success_read;
	//Open the weights files
	FILE* file = fopen("weights.bin", "rb");
	//Allocate memory for all weights
	LENET_T* weights = (LENET_T*)malloc(61470 * sizeof(LENET_T));
	float* temp = (float*)malloc(61470 * sizeof(float));
	//Check if the weights array is allocated and the weights files is open
	if (!file || !weights || !temp){
		return NULL;
	}
	//Read the weights
	success_read = fread(temp, sizeof(float), 61470, file);
	if(success_read != 61470)
		return NULL;
	for (int i = 0; i < 61470; i++){
		weights[i] = temp[i];
	}
	free(temp);
	//Return the weights
	return weights;
}


/**
 * Read biases from a binary file
 *
 * @return: array with all bias read
 *
*/
LENET_T* read_bias() {
	//Variable for checking
	size_t success_read;
	//Open the bias file
	FILE* file = fopen("bias.bin", "rb");
	//Allocate memory for bias array
	LENET_T* bias = (LENET_T*)malloc(236 * sizeof(LENET_T));
	float* temp = (float*)malloc(236 * sizeof(float));
	//Check if the bias array was successfully allocated and if the bias files is open
	if (!file || !bias || !temp){
		return NULL;
	}
	//Read all bias fom file
	success_read = fread(temp, sizeof(LENET_T), 236, file);
	if(success_read != 236)
		return NULL;
	for (int i = 0; i < 236; i++){
		bias[i] = temp[i];
	}
	//Returne the bias array
	return bias;
}


/**
 * Allocatde a 3D tensor
 *
 * @param[in]: number of samples needed to allocate memory
 * @return: Allocated memory allocated
 *
*/
LENET_T*** allocate_images(int number_of_images){
	//Allocate an array to store the images
	LENET_T*** tensor = (LENET_T***)malloc(60000 * sizeof(LENET_T**));
	for (int i = 0; i < 60000; i++){
		//Allocate memory for the image rows
		tensor[i] = (LENET_T**)malloc(28 * sizeof(LENET_T*));
		for (int j = 0; j < 28; j++){
			//Allocate memory for the image cols
			tensor[i][j] = (LENET_T*)malloc(28 * sizeof(LENET_T));
		}
	}
	return tensor;
}

int ReverseInt (int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

/**
 * Read the dataset from the binary file
 *
 * @param[in]: number of samples to read from the dataset
 * @return: Tensor with the dataset
 *
*/

LENET_T*** ReadMNIST(int num_images_read) {
	std::ifstream file ("train-images-idx3-ubyte", std::ios::binary);
	if (file.is_open()) {
		LENET_T*** images = allocate_images(num_images_read);
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
		for(int i = 0; i < num_images_read; ++i)
		{
			for(int r = 0; r < 28; ++r)
			{
				for(int c = 0; c < 28; ++c)
				{
					unsigned char temp=0;
					file.read((char*)&temp,sizeof(temp));
					images[i][r][c]= ((float)temp)/255;
				}
			}
		}
		return images;
	}
	else{
		std::cout << "Data set file not found" << std::endl;
		return NULL;
	}
}



int main(){

	//Variables definition;
	LENET_T* weights;
	LENET_T* bias;
	LENET_T*** dataset;
	
	LENET_T input[28][28];
	LENET_T output[10];


	//Read LeNet parameters
	weights = read_weights();
	bias =  read_bias();
	//Read dataset
	dataset = ReadMNIST(60000);
	//Check that the parameters have been read correctly
	if (!weights || !bias || !dataset){
		printf("Problem in reading parameter files\n");
		return 1;
	}
	//Write the dataset sample into the input array
	for (int i = 0; i < 28; i++){
		for (int j = 0; j < 28; j++){
			input[i][j] = dataset[0][i][j];
		}
	}
	//Execute the classification using LeNet
	lenet(weights, bias, input, output);
	//Print the output
	for (int i = 0; i < 10; i++){
		std::cout << output[i] << std::endl;
	}

	return 0;
}
