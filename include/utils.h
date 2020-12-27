#ifndef UTILS_H_
#define UTILS_H_

#include <ap_axi_sdata.h>
#include <hls_stream.h>

typedef ap_axiu<32,4,5,5> AXI_VALUE;
typedef float LENET_T;

/**
 * Add zero padding on the input
 *
 * @param[in] input:	Input image
 * @param[out] output:	Padded image
 *
*/
void padding (LENET_T in[28][28], LENET_T out[32][32]);

/**
 * Calculate the activation value using the ReLU function
 *
 * @param[in] input:	Input value
 * @param[out] output:	Activation value
 *
*/
LENET_T relu_activation (LENET_T input);

/**
 * Read weigths and biases of convolution layers
 *
 * @param[in] kernel_num: number of kernels on the convolutional layer
 * @param[in] kernel_size: kernel size
 * @param[in] feature_maps_input: number of channels of the input image
 * @param[in] weights_stream: stream used to send the weights
 * @param[in] bias_stream: stream used to send the weights
 * @param[out] weights_buffer: buffer to store all weigths read
 * @param[out] bias_buffer: buffer to store all biases read
*/
void read_params_conv (int kernel_num, int kernel_size, int feature_maps_input, hls::stream<AXI_VALUE>& weights_stream,
					   hls::stream<AXI_VALUE>& bias_stream, LENET_T* weights_buffer, LENET_T* bias_buffer);

/**
 * Read weights and biases of dense layers
 *
 * @param[in] input_size: number of elements on the input array
 * @param[in] neurons_num: number of neurons in the layer
 * @param[in] weights_stream: stream used to send the weights
 * @param[in] bias_stream: stream used to send the weights
 * @param[out] weights_buffer: buffer to store all weigths read
 * @param[out] bias_buffer: buffer to store all biases read
*/
void read_params_dense (int input_size, int neurons_num, hls::stream<AXI_VALUE>& weights_stream,
		                hls::stream<AXI_VALUE>& bias_stream, LENET_T* weights_buffer, LENET_T* bias_buffer);

/**
 * Read the input image and store into an intern buffer
 *
 * @param[in] image: stream with the input image
 * @param[out] image_buffer: internal buffer to store the image
*/
void read_input(hls::stream<AXI_VALUE>& image, LENET_T image_buffer[28][28]);

/**
 * Write the output into a stream
 *
 * @param[in] lenet_out: LeNet output
 * @param[out] out_stream: output stream
*/
void write_output(LENET_T lenet_out[10], hls::stream<AXI_VALUE>& out_stream);


#endif
