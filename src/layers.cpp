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
				output[r][c][f] = relu_activation(acc);
			}
		}
	}
}
