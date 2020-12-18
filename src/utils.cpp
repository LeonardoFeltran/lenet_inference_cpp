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


void relu_activation(LENET_T input, LENET_T output){
	output = input < 0 ? 0 : input;
}
