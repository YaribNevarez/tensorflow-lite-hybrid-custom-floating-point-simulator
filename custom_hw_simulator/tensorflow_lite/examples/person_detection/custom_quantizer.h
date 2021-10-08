#ifndef CUSTOM_QUANTIZER_H_
#define CUSTOM_QUANTIZER_H_

#include "stddef.h"

namespace custom_float
{

typedef enum
{
  NONE,
  CUSTOM_150,
  CUSTOM_151,
  CUSTOM_152,
  CUSTOM_153,
  CUSTOM_154,
  FP16,
  BFLOAT16,
  TENSOR_FLOAT,
  FP32
} CustomFloatType;

typedef struct TensorQuantization
{
  CustomFloatType type;
  int sign;
  int exponent;
  int mantissa;
  int accuracy;
};

void TensorQuantizer_setTensorID (int id);

const char * TensorQuantizer_getTypeName (int id);

void TensorQuantizer_setType (CustomFloatType type);

float * TensorQuantizer_quantize (const float * tensor, size_t size);

}

#endif  // CUSTOM_QUANTIZER_H_
