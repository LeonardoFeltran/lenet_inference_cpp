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
 * Read weights from a stream and store them into a tensor
 *
 * @param[in] weights_stream:	Stream with all weights queued 
 * @param[out] weights_tensor: 	Tensor to store all weights
*/
void read_weights (LENET_T* weights_stream, LENET_T weights_tensor[5][5][1][6]);

/**
 * Read all biases from a stream and store them into a tensor
 *
 * @param[in] bias_stream:	Stream with all biases queued 
 * @param[out] bias_tensor: Tensor to store all biases
*/
void read_biases (LENET_T* weights_stream, LENET_T weights_tensor[6]);


#endif
