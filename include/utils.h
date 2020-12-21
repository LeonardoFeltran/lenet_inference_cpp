#ifndef UTILS_H_
#define UTILS_H_

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
void read_params_conv (int kernel_num, int kernel_size, int feature_maps_input, LENET_T weights_stream[61470],
					   LENET_T bias_stream[236], LENET_T* weights_buffer, LENET_T* bias_buffer);

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
void read_params_dense (int input_size, int neurons_num, LENET_T weights_stream[61470], 
						LENET_T bias_stream[236], LENET_T* weights_buffer, LENET_T* bias_buffer);



#endif
