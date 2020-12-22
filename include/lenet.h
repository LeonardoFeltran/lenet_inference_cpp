#ifndef LENET_H_
#define LENET_H_

/**
 * LeNet architecture used for inference
 *
 * @param[in] weights:		All pre-trained weights used by the architecture
 * @param[in] bias:			All pre-trained biases
 * @param[in] input:		Sample to be classified (2D matrix)
 * @param[out] output:		The output value for each class
 *
*/
void lenet(LENET_T weights[61470], LENET_T bias[236], LENET_T input[28][28], LENET_T output[10]);

#endif
