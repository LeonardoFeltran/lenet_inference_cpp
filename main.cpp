#include <iostream>
#include <fstream>
#include <hls_stream.h>
#include <typeinfo>

#include "include/utils.h"
#include "include/layers.h"
#include "include/lenet.h"


/**
 * Read weights from file
 *
 * @return: array with all weights read
 *
*/
void read_weights(hls::stream<AXI_VALUE>& weights) {
	//AXI variable
	AXI_VALUE aValue;
	//Variable for checking
	size_t success_read;
	//Open the weights files
	FILE* file = fopen("weights.bin", "rb");
	//Allocate memory for all weights
	float* temp = (float*)malloc(61470 * sizeof(float));
	//Check if the weights array is allocated and the weights files is open
	if (!file || !temp){
		return;
	}
	//Read the weights
	success_read = fread(temp, sizeof(float), 61470, file);
	//Check if all weights were successfully read
	if(success_read != 61470)
		return;
	for (int i = 0; i < 61470; i++){
		//Conversion to get the bit representation
		union {	unsigned int oval; float ival; } converter;
		converter.ival = temp[i];
		aValue.data = converter.oval;
		//Put the data in the stream
		weights.write(aValue);
	}
	free(temp);
}


/**
 * Read biases from a binary file
 *
 * @return: array with all bias read
 *
*/
void read_bias(hls::stream<AXI_VALUE>& bias) {
	//AXI variable
	AXI_VALUE aValue;
	//Variable for checking
	size_t success_read;
	//Open the bias file
	FILE* file = fopen("bias.bin", "rb");
	//Allocate memory to store all bias read
	float* temp = (float*)malloc(236 * sizeof(float));
	//Check if the bias array was successfully allocated and if the bias files is open
	if (!file || !temp){
		return;
	}
	//Read all bias fom file
	success_read = fread(temp, sizeof(LENET_T), 236, file);
	//Check if all weights were successfully read
	if(success_read != 236)
		return;
	for (int i = 0; i < 236; i++){
		//Conversion to get the bit representation
		union {	unsigned int oval; float ival; } converter;
		converter.ival = temp[i];
		aValue.data = converter.oval;
		//Put the data in the stream
		bias.write(aValue);
	}
	free(temp);
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
	 LENET_T*** dataset;
	 AXI_VALUE aValue;
	 LENET_T temp;
	 //Stream for data transfer
	 hls::stream<AXI_VALUE> weights;
	 hls::stream<AXI_VALUE> bias;
	 hls::stream<AXI_VALUE> input;
	 hls::stream<AXI_VALUE> output;
	//Read LeNet parameters
	read_weights(weights);
	read_bias(bias);
	//Read dataset
	dataset = ReadMNIST(60000);
	//Check that the parameters have been read correctly
	if (!dataset ){
		printf("Problem in reading parameter files\n");
		return 1;
	}
	//Write the dataset sample into the input array
	for (int i = 0; i < 28; i++){
		for (int j = 0; j < 28; j++){
			union {	unsigned int oval; float ival; } converter;
			converter.ival = dataset[0][i][j];
			aValue.data = converter.oval;
			input.write(aValue);
		}
	}
	//Execute the classification using LeNet
	lenet(weights, bias, input, output);
	//Print the output
	for (int i = 0; i < 10; i++){
		//Read the output
		output.read(aValue);
		//Converte the data read to float
		union {	unsigned int ival; float oval; } converter;
		converter.ival = aValue.data;
		std::cout << converter.oval << std::endl;
	}
	return 0;
}
