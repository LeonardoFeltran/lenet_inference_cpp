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

#endif
