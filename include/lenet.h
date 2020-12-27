#ifndef LENET_H_
#define LENET_H_

/**
 * LeNet architecture used for inference
 *
 * @param[in] weights:		Stream of weights
 * @param[in] bias:			Stream of biases
 * @param[in] image:		Stream with input image
 * @param[out] out_stream:	Output stream
 *
*/
void lenet(hls::stream<AXI_VALUE>& weights, hls::stream<AXI_VALUE>& bias,
		   hls::stream<AXI_VALUE>& image, hls::stream<AXI_VALUE>& out_stream);

#endif
