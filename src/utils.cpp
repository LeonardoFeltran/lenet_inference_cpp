#include "../include/utils.h"


void padding (LENET_T in[28][28], LENET_T out[32][32]) {
	Row: for (int r = 0; r < 32; r++) {
		Col: for (int c = 0; c < 32; c++) {
			//Copy the input pixel if the position is out of the padding region
			if(c > 1 and r > 1 and c < 30 and r < 30)
				out[r][c] = in[r - 2][c - 2];
			//Put zero if the position is on the padding region
			else
				out[r][c] = 0;
		}
	}
}


LENET_T relu_activation(LENET_T input){
	// Check if the input id negative
	// If true, the output is zero else it receives the input
	return input < 0 ? (LENET_T) 0 : input;
}

void read_params_conv (int kernel_num, int kernel_size, int feature_maps_input,
					  LENET_T weights_stream[61470], LENET_T bias_stream[236],
					   LENET_T* weights_buffer, LENET_T* bias_buffer){
	//Calculate the number of weigths to read
	int num_weights = kernel_num * kernel_size * kernel_size * feature_maps_input;
	//Reads all weights and store on internal buffers
	Read_weights_conv: for (int i = 0; i < num_weights; i++){
		weights_buffer[i] = weights_stream[i];
	}
	//Reads all biases and store on internal buffers
	Read_bias_conv: for (int p = 0; p < kernel_num; p++){
		bias_buffer[p] = bias_stream[p];
	}
}

void read_params_dense (int input_size, int neurons_num, LENET_T weights_stream[61470], LENET_T bias_stream[236],
					   LENET_T* weights_buffer, LENET_T* bias_buffer){
	//Address to read and store the values
	int num_weights = input_size * neurons_num;
	//Reads all weights and store on internal buffers
	Read_weights_dense: for (int i = 0; i < num_weights; i++){
		weights_buffer[i] = weights_stream[i];
	}
	//Reads all biases and store on internal buffers
	Read_bias_dense: for (int p = 0; p < neurons_num; p++){
		bias_buffer[p] = bias_stream[p];
	}
}
