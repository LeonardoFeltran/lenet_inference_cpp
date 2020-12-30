#include "../include/layers.h"


void conv1 (LENET_T weights[150], LENET_T bias[6],
			LENET_T input[28][28], LENET_T output[28][28][6]) {
	#pragma HLS INLINE region
	//Temporary input for padded input
	LENET_T input_temp[32][32];
	#pragma HLS ARRAY_PARTITION variable=input_temp cyclic factor=5 dim=2
	#pragma HLS ARRAY_PARTITION variable=weights block factor=25 dim=1
	//Add padding at the input
	padding(input, input_temp);
	//Filter position
	Row: for (int r = 0; r < 28; r++) {
		Col: for (int c = 0; c < 28; c++) {
			//Select the filter
			Filter: for (int f = 0; f < 6; f++) {
			#pragma HLS PIPELINE
				//Accumulator to store weights and input multiplication
				LENET_T acc = 0;
				//Multiply the input patch and weights of filter f
				Filter_row: for (int fr = 0; fr < 5; fr++) {
					Filter_col: for (int fc = 0; fc < 5; fc++) {
						acc += weights[fr*30 + fc*6 + f] * input_temp[r + fr][c + fc];
					}
				}
				//Acitvation function
				acc += bias[f];
				output[r][c][f] = relu_activation(acc);
			}
		}
	}
}

void maxPooling1(LENET_T input[28][28][6], LENET_T output[14][14][6]) {
	//Temporary variables
	LENET_T max;
	LENET_T temp;
	//Iterates over the input tensor
	Feature_map: for (int fm = 0; fm < 6; fm++) {
		Rows: for (int r = 0; r < 28; r += 2) {
		#pragma HLS LOOP_FLATTEN off
			Cols: for (int c = 0; c < 28; c += 2) {
			#pragma HLS LOOP_FLATTEN off
				max = 0;
				//Iterates over the filter to get the maximum value
				Kernel_row: for (int fr = 0; fr < 2; fr++) {
					Kernel_col: for (int fc = 0; fc < 2; fc++) {
					#pragma HLS PIPELINE
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

void conv2 (LENET_T weights[2400], LENET_T bias[16],
			LENET_T input[14][14][6], LENET_T output[10][10][16]) {
	#pragma HLS ARRAY_PARTITION variable=input cyclic factor=5 dim=2
	#pragma HLS ARRAY_PARTITION variable=weights block factor=30 dim=1
	//Iterate over the input
	Row: for (int r = 0; r < 10; r++) {
		Col: for (int c = 0; c < 10; c++) {
			//Select the filter
			Filter: for (int f = 0; f < 16; f++) {
				//Accumulator to store weights and input multiplication
				LENET_T acc = 0;
				//Multiply the input patch and weights of filter f
				Feature_map: for (int fm = 0; fm < 6; fm++) {
				#pragma HLS PIPELINE II=9
					Filter_row: for (int fr = 0; fr < 5; fr++) {
						Filter_col: for (int fc = 0; fc < 5; fc++) {
							acc += weights[fr*480 + fc*96 + fm*16 + f] * input[r + fr][c + fc][fm];
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
	//Temporary variables
	LENET_T max;
	LENET_T temp;
	//Iterates over the input tensor
	Feature_map: for (int fm = 0; fm < 16; fm++)	{
		Rows: for (int r = 0; r < 10; r += 2) {
			Cols: for (int c = 0; c < 10; c += 2) {
			#pragma HLS LOOP_FLATTEN off
				max = 0;
				//Iterates over the filter to get the maximum value
				Kernel_row: for (int kr = 0; kr < 2; kr++) {
					Kernel_col: for (int kc = 0; kc < 2; kc++) {
					#pragma HLS PIPELINE
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

void conv3(LENET_T weights[48000], LENET_T bias[120],
		   LENET_T input[5][5][16], LENET_T output[120]) {
	#pragma HLS ARRAY_PARTITION variable=input cyclic factor=5 dim=2
	#pragma HLS ARRAY_PARTITION variable=weights block factor=5 dim=1
	//Iterate over the input
	Filter: for (int f = 0; f < 120; f++) {
	#pragma HLS LOOP_FLATTEN off
		//Accumulator of matrix multiplication
		LENET_T acc = 0;
		//Multiply all filter weights with all patch values
		Feature_map: for (int fm = 0; fm < 16; fm++) {
		#pragma HLS PIPELINE II=9
			Kernel_row: for (int kr = 0; kr < 5; kr++) {
				Kernel_col: for (int kc = 0; kc < 5; kc++) {
					acc += weights[kr*9600 + kc*1920 + fm*120 + f] * input[kr][kc][fm];
				}
			}
		}
		//Add bias and write output
		acc += bias[f];
		output[f] = relu_activation(acc);		
	}
}

void dense1(LENET_T weights[10080], LENET_T bias[84],
		    LENET_T input[120], LENET_T output[84]) {
	//Temporary variables
	LENET_T acc;
	//Dot product between the input array and weights matrix
	Row_d1: for (int r = 0; r < 84; r++) {
		acc = 0;
		Col_d1: for (int c = 0; c < 120; c++) {
		#pragma HLS PIPELINE
			acc += input[c] * weights[c*84 + r];
		}
		//Add bias and write output
		acc += bias[r];
		output[r] = relu_activation(acc);
	}
}


void dense2(LENET_T weights[840], LENET_T bias[10],
			LENET_T input[84], LENET_T output[10]) {
	//Temporary variables
	LENET_T acc;
	//Dot product between the input array and weights matrix
	Row_d2: for (int r = 0; r < 10; r++) {
		acc = 0;
		Col_d2: for (int c = 0; c < 84; c++) {
		#pragma HLS PIPELINE
			acc += input[c] * weights[c*10 + r];
		}
		//Add bias and write output
		output[r] = acc + bias[r];
	}
}
