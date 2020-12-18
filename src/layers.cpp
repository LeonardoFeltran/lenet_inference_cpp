#include "../include/layers.h"


void conv1 (LENET_T weights[5][5][1][6], LENET_T bias[6],
			LENET_T input[28][28], LENET_T output[28][28][6]) {
	//Temporary input for padded input
	LENET_T input_temp[32][32];
	//Add padding at the input
	padding(input, input_temp);
	//Filter position
	Row: for (int r = 0; r < 28; r++) {
		Col: for (int c = 0; c < 28; c++) {
			//Select the filter
			Filter: for (int f = 0; f < 6; f++) {
				//Accumulator to store weights and input multiplication
				LENET_T acc = 0;
				//Multiply the input patch and weights of filter f
				Filter_row: for (int fr = 0; fr < 5; fr++) {
					Filter_col: for (int fc = 0; fc < 5; fc++) {
						acc += weights[fr][fc][0][f] * input_temp[r + fr][c + fc];
						
					}
				}
				acc += bias[f];
				//Acitvation function
				output[r][c][f] = relu_activation(acc);
			}
		}
	}
}

void maxPooling1(LENET_T input[28][28][6], LENET_T output[14][14][6]) {
	//Internal vairables declaration
	LENET_T max;
	LENET_T temp;
	//Iterates over the input tensor
	Feature_map: for (int fm = 0; fm < 6; fm++) {
		Rows: for (int r = 0; r < 28; r += 2) {
			Cols: for (int c = 0; c < 28; c += 2) {
				max = 0;
				//Iterates over the filter to get the maximum value
				Kernel_row: for (int fr = 0; fr < 2; fr++) {
					Kernel_col: for (int fc = 0; fc < 2; fc++) {
						temp = input[r + fr][c + fc][fm];
						if(temp > max)
							max = temp;
					}
				}
				//Save the maximum value on the output tensor
				output[r/2][c/2][fm] = max;
			}
		}
	}
}

void conv2 (LENET_T weights[5][5][6][16], LENET_T bias[16],
			LENET_T input[14][14][6], LENET_T output[10][10][16]) {

	//Iterate over the input
	Row: for (int r = 0; r < 10; r++) {
		Col: for (int c = 0; c < 10; c++) {
			//Select the filter
			Filter: for (int f = 0; f < 16; f++) {
				//Accumulator to store weights and input multiplication
				LENET_T acc = 0;
				//Multiply the input patch and weights of filter f
				Feature_map: for (int fm = 0; fm < 6; fm++) {
					Filter_row: for (int fr = 0; fr < 5; fr++) {
						Filter_col: for (int fc = 0; fc < 5; fc++) {
							acc += weights[fr][fc][fm][f] * input[r + fr][c + fc][fm];
						}
					}
				}
				//Store the calculated value in the output
				acc += bias[f];
				output[r][c][f] = relu_activation(acc);
			}
		}
	}
}

void maxPooling2(LENET_T input[10][10][16], LENET_T output[5][5][16]) {
	//Internal vairables declaration
	LENET_T max;
	LENET_T temp;
	//Iterates over the input tensor
	Feature_map: for (int fm = 0; fm < 16; fm++)	{
		Rows: for (int r = 0; r < 10; r += 2) {
			Cols: for (int c = 0; c < 10; c += 2) {
				max = 0;
				//Iterates over the filter to get the maximum value
				Kernel_row: for (int kr = 0; kr < 2; kr++) {
					Kernel_col: for (int kc = 0; kc < 2; kc++) {
						temp = input[r + kr][c + kc][fm];
						if(temp > max)
							max = temp;
					}
				}
				//Save the maximum value on the output tensor
				output[r/2][c/2][fm] = max;
			}
		}
	}
}