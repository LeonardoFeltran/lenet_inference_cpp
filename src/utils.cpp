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
	return input < 0 ? (LENET_T)0 : input;
}

void read_weights (LENET_T* weights_stream, LENET_T weights_tensor[5][5][1][6]){
	//Iterate over the output tensor to write all values from the input
	Kernel_row: for (int kr = 0; kr < 5; kr++){
		Kernel_col: for (int kc = 0; kc < 5; kc++){
			I_Feature_Map: for (int ifm = 0; ifm < 1; ifm++){
				O_Feature_Map: for (int ofm = 0; ofm < 6; ofm++){
					weights_tensor[kr][kc][ifm][ofm] = weights_stream[ofm + ifm*6 * kc*30 + kr*150];
				}
			}
		}
	}
}

void read_biases (LENET_T* weights_stream, LENET_T weights_tensor[6]){
	Read_bias: for (int i = 0; i < 6; i++){
		weights_tensor[i] = weights_stream[i];
	}
}
