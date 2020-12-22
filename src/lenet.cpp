#include "../include/layers.h"
#include "../include/utils.h"



void lenet(LENET_T weights[61470], LENET_T bias[236], LENET_T input[28][28], LENET_T output[10]){
	#pragma HLS INLINE off
	#pragma HLS PIPELINE II=1

	//Arguments for the first convolutional layer
	LENET_T conv1_weights[150];
	LENET_T	conv1_bias[6];
	LENET_T conv1_out[28][28][6];
	//Arguments for the second convolutional layer
	LENET_T conv2_weights[2400];
	LENET_T	conv2_bias[16];
	LENET_T conv2_in[14][14][6];	
	LENET_T conv2_out[10][10][16];
	//Arguments for the third convolutional layer
	LENET_T conv3_weights[48000];
	LENET_T conv3_bias[120];
	LENET_T conv3_in[5][5][16];
	LENET_T conv3_out[120];
	//Arguments for the first dense layer
	LENET_T dense1_weights[10080];
	LENET_T dense1_bias[84];
	LENET_T dense1_out[84];
	//Arguments for the second dense layer
	LENET_T dense2_weights[840];
	LENET_T dense2_bias[10];

	//Read parameters (weights and biases) for the first convolutional layer
	read_params_conv(6, 5, 1, weights, bias, conv1_weights, conv1_bias);
	//First convolutional layer with ReLU activation
	conv1(conv1_weights, conv1_bias, input, conv1_out);
	//Max pooling layer for dimension reduction and get more relevant features
	maxPooling1(conv1_out, conv2_in);

	//Read parameters (weights and biases) for the second convolutional layer
	read_params_conv(16, 5, 6, weights + 150, bias + 6, conv2_weights, conv2_bias);
	//Second convolutional layer with ReLU activation
	conv2(conv2_weights, conv2_bias, conv2_in, conv2_out);
	//Max pooling layer for dimension reduction and get more relevant features
	maxPooling2(conv2_out, conv3_in);

	//Read parameters (weights and biases) for the third convolutional layer
	read_params_conv(120, 5, 16, weights + 2550, bias + 22, conv3_weights, conv3_bias);
	//Third convolutional layer with ReLU activation	
	conv3(conv3_weights, conv3_bias, conv3_in, conv3_out);
	
	//Read parameters (weights and biases) for the first dense layer
	read_params_dense(120, 84, weights + 50550, bias + 142, dense1_weights, dense1_bias);
	//First dense layer with ReLU activation	
	dense1(dense1_weights, dense1_bias, conv3_out, dense1_out);

	//Read parameters (weights and biases) for the second dense layer
	read_params_dense(84, 10, weights + 60630, bias + 226, dense2_weights, dense2_bias);
	//Second dense layer with linear activation	
	dense2(dense2_weights, dense2_bias, dense1_out, output);

}

