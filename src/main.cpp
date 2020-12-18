#include <iostream>


/**
 * Read weights from file
 *
 * @return: array with all weights read
 *
*/
LENET_T* read_weights () {
	//Open the weights files
	FILE* file = fopen("weights.bin", "rb");
	//Allocate memory for all weights
	LENET_T* weights = (LENET_T*)malloc(61470 * sizeof(LENET_T));
	//Check if the weights array is allocated and the weights files is open
	if (!file || !weights){
		return NULL;
	}
	//Read the weights
	fread(weights, sizeof(LENET_T), 61470, f);
	//Return the weights
	return weights;
}


/**
 * Read biases from a binary file
 *
 * @return: array with all bias read
 *
*/
LENET_T* read_bias () {
	//Open the bias file
	FILE* file = fopen("bias.bin", "rb");
	//Allocate memory for bias array
	LENET_T* bias = (LENET_T*)malloc(236 * sizeof(LENET_T));
	//Check if the bias array was succecfully acllocated and if the bias files is open
	if (!file || !bias){
		return NULL;
	}
	//Read all bias fom file
	fread(bias, sizeof(LENET_T), 236, f);
	//Returne the bias array
	return bias;
}

int main(){
	
	
	
	return 0;
}