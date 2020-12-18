#ifndef UTILS_H_
#define UTILS_H_

typedef float LENET_T;

/**
 * Add zero padding on the input
 *
 * @param[in] input:		Input image
 * @param[out] output:		Padded image
 *
*/
void padding(LENET_T in[28][28], LENET_T out[32][32]);

/**
 * Calculate the activation value using the ReLU function
 *
 * @param[in] input:		Input value
 * @param[out] output:		Activation value
 *
*/
void relu_activation(LENET_T input, LENET_T output);

#endif
