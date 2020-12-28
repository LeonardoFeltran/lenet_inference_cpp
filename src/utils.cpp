#include "../include/utils.h"


void padding (LENET_T in[28][28], LENET_T out[32][32]) {
	#pragma HLS INLINE region
	Row: for (int r = 0; r < 32; r++) {
		Col: for (int c = 0; c < 32; c++) {
		#pragma HLS PIPELINE
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

void read_params_conv (int kernel_num, int kernel_size, int feature_maps_input, hls::stream<AXI_VALUE>& weights_stream,
					   hls::stream<AXI_VALUE>& bias_stream, LENET_T* weights_buffer, LENET_T* bias_buffer){
	//Temporary variable
	AXI_VALUE aValue;
	//Calculate the number of weights to read
	int num_weights = kernel_num * kernel_size * kernel_size * feature_maps_input;
	//Reads all weights and store on internal buffers
	Read_weights_conv: for (int i = 0; i < num_weights; i++){
		//Read the input
		weights_stream.read(aValue);
		/***** FOR LENET_T == float *****
		//Convert the data
		union {	unsigned int ival; float oval; } converter;
		converter.ival = aValue.data;
		weights_buffer[i] = converter.oval;
		*/
		weights_buffer[i].range() = aValue.data;
	}
	//Reads all biases and store on internal buffers
	Read_bias_conv: for (int p = 0; p < kernel_num; p++){
		//Read the input
		bias_stream.read(aValue);
		/***** FOR LENET_T == float *****
		union {	unsigned int ival; float oval; } converter;
		converter.ival = aValue.data;
		bias_buffer[p] = converter.oval;
		*/
		bias_buffer[p].range() = aValue.data;
	}
}

void read_params_dense (int input_size, int neurons_num, hls::stream<AXI_VALUE>& weights_stream,
		                hls::stream<AXI_VALUE>& bias_stream, LENET_T* weights_buffer, LENET_T* bias_buffer){
	//Temporary variable
	AXI_VALUE aValue;
	//Address to read and store the values
	int num_weights = input_size * neurons_num;
	//Reads all weights and store on internal buffers
	Read_weights_dense: for (int i = 0; i < num_weights; i++){
		//Read the input
		weights_stream.read(aValue);
		/***** FOR LENET_T == float *****
		//Convert the data
		union {	unsigned int ival; float oval; } converter;
		converter.ival = aValue.data;
		weights_buffer[i] = converter.oval;
		*/
		weights_buffer[i].range() = aValue.data;
	}
	//Reads all biases and store on internal buffers
	Read_bias_dense: for (int p = 0; p < neurons_num; p++){
		//Read the input
		bias_stream.read(aValue);
		/***** FOR LENET_T == float *****
		//Convert the data
		union {	unsigned int ival; float oval; } converter;
		converter.ival = aValue.data;
		bias_buffer[p] = converter.oval;
		*/
		bias_buffer[p].range() = aValue.data;
	}
}

void read_input(hls::stream<AXI_VALUE>& image, LENET_T image_buffer[28][28]){
	//Temporary variable
	AXI_VALUE aValue;
	for (int i = 0; i < 28; i++){
		for (int j = 0; j < 28; j++){
			//Read the input
			image.read(aValue);
			/***** FOR LENET_T == float *****
			//Convert the data
			union {	unsigned int ival; float oval; } converter;
			converter.ival = aValue.data;
			image_buffer[i][j] = converter.oval;
			*/
			image_buffer[i][j].range() = aValue.data;
		}
	}
}

void write_output(LENET_T lenet_out[10], hls::stream<AXI_VALUE>& out_stream){
	//Temporary variable
	AXI_VALUE aValue;
	for (int i = 0; i < 10; i++){
		/***** FOR LENET_T == float *****
		//Convert the output
		union {	unsigned int oval; float ival; } converter;
		converter.ival = lenet_out[i];
		aValue.data = converter.oval;
		*/
		aValue.data = lenet_out[i].range();
		//Write side channel signals
		aValue.last = (i == 9) ? 1 : 0;
		aValue.strb = -1;
		aValue.keep = 15;
		aValue.user = 0;
		aValue.id = 0;
		aValue.dest = 0;
		//Write the data
		out_stream.write(aValue);
	}
}
