#ifndef WAV_DEBUG_H
#define WAV_DEBUG_H

#include "tensor.h"

void print(const Tensor &tensor, int n, int *idx);
void print(const Tensor &tensor, const string &filename = "cout");

#endif
