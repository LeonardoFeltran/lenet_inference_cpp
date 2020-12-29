#ifndef LAYERS_H_
#define LAYERS_H_

#include <iostream>
#include <cmath>

#include "../include/utils.h"


/**
 * Convolute over the input image
 * 
 * @param[in] weights:		matrix with all kernel's weights
 * @param[in] bias:			bias for each kernel
 * @param[in] input:		input image to be convoluted
 * @param[out] output:		features maps extracted by the layer
 * 
*/
void conv1(LENET_T weights[150], LENET_T bias[6], LENET_T input[28][28], LENET_T output[28][28][6]);


/**
 * Convolute over the input image
 *
 * @param[in] weights:		matrix with all kernel's weights
 * @param[in] bias:			bias for each kernel
 * @param[in] input:		input image to be convoluted
 * @param[out] output:		features maps extracted by the layer
 *
*/
void conv2(LENET_T weights[2400], LENET_T bias[16], LENET_T input[14][14][6], LENET_T output[10][10][16]);


/**
 * Convolute over the input image
 *
 * @param[in] weights:		matrix with all kernel's weights
 * @param[in] bias:			bias for each kernel
 * @param[in] input:		input image to be convoluted
 * @param[out] output:		features maps extracted by the layer
 *
*/
void conv3(LENET_T weights[48000], LENET_T bias[120], LENET_T input[5][5][16], LENET_T output[120]);


/**
 * Activation function ReLU for dense layer
 *
 * @param[in/out] arr:		Array with output of previous layer
 *
*/
void relu_linear2(LENET_T input[84], LENET_T output[84]);


/**
 * Downsamples the input by taking the maximum value over a kernel of size 2x2 and stride 2
 *
 * @param[in] input:		Output of previous layer
 * @param[out] output:		Downsampled data
 *
*/
void maxPooling1(LENET_T input[28][28][6], LENET_T output[14][14][6]);


/**
 * Downsamples the input by taking the maximum value over a kernel of size 2x2 and stride 2
 *
 * @param[in] input:		Output of previous layer
 * @param[out] output:		Downsampled data
 *
*/
void maxPooling2(LENET_T input[10][10][16], LENET_T output[5][5][16]);


/**
 * Layer with artificial neurons for classification without activation function
 *
 * @param[in] weights:		Weights of all neurons of this layer
 * @param[in] bias:			Bias of all neurons of this layer
 * @param[in] input:		Output of previous layer
 * @param[out] output:		Layers activation pattern
 *
*/
void dense1(LENET_T* weights, LENET_T* bias,
			LENET_T input[120], LENET_T output[84]);


/**
 * Layer with artificial neurons for classification without activation function
 *
 * @param[in] weights:		Weights of all neurons of this layer
 * @param[in] bias:			Bias of all neurons of this layer
 * @param[in] input:		Output of previous layer
 * @param[out] output:		Layers activation pattern
 *
*/
void dense2(LENET_T* weights, LENET_T* bias,
			LENET_T input[84], LENET_T output[10]);


void test(LENET_T* weights, LENET_T* bias, LENET_T input[28][28], LENET_T output[28][28][6]);

#endif
