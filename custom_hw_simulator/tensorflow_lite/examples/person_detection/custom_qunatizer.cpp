#include "custom_quantizer.h"
#include "stdint.h"

namespace custom_float
{

  typedef int32_t         CustomFormat;
  typedef int64_t         MagnitudeFormat;
  typedef int8_t          ExponentFormat;
  typedef bool            SignFormat;

static  const char * quantization_name[] =
{
    "NONE",
    "CUSTOM_150",
    "CUSTOM_151",
    "CUSTOM_152",
    "CUSTOM_153",
    "CUSTOM_154",
    "FP16",
    "BFLOAT16",
    "TENSOR_FLOAT",
    "FP32"
};

static TensorQuantization quantizations[] =
{
    {
        .type = CUSTOM_150,
        .sign = 1,
        .exponent = 5,
        .mantissa = 0,
        .accuracy = 0
    },
    {
        .type = CUSTOM_151,
        .sign = 1,
        .exponent = 5,
        .mantissa = 1,
        .accuracy = 0
    },
    {
        .type = CUSTOM_152,
        .sign = 1,
        .exponent = 5,
        .mantissa = 2,
        .accuracy = 0
    },
    {
        .type = CUSTOM_153,
        .sign = 1,
        .exponent = 5,
        .mantissa = 3,
        .accuracy = 0
    },
    {
        .type = CUSTOM_154,
        .sign = 1,
        .exponent = 5,
        .mantissa = 4,
        .accuracy = 0
    },
    {
        .type = FP16,
        .sign = 1,
        .exponent = 5,
        .mantissa = 10,
        .accuracy = 0
    },
    {
        .type = BFLOAT16,
        .sign = 1,
        .exponent = 8,
        .mantissa = 7,
        .accuracy = 0
    },
    {
        .type = TENSOR_FLOAT,
        .sign = 1,
        .exponent = 8,
        .mantissa = 10,
        .accuracy = 0
    },
    {
        .type = FP32,
        .sign = 1,
        .exponent = 8,
        .mantissa = 23,
        .accuracy = 0
    }
};

static float tensors[16][64 * 1024] = { 0 };
static CustomFloatType type_array[16] = { NONE };
static CustomFloatType target_type = NONE;

static int tensor_id = 0;


void TensorQuantizer_setTensorID (int id)
{
  tensor_id = id;
}

void TensorQuantizer_setType (CustomFloatType type)
{
  target_type = type;
}

#define DATA32_GET_EXPONENT(x) ((0xFF & ((x) >> 23)) - 0x7F)
#define DATA32_GET_MANTISSA(x) (0x00800000 | ((0x7FFFFF) & (x)))
#define DATA32_GET_SIGN(x)     (0x80000000 & (x))

#define CUSTOM_GET_SIGN(x, CUSTOM_EXPONENT_BIT_WIDTH, CUSTOM_MANTISSA_BIT_WIDTH)            ((1 << ((CUSTOM_EXPONENT_BIT_WIDTH)+(CUSTOM_MANTISSA_BIT_WIDTH))) & (x))
#define CUSTOM_GET_EXPONENT(x, CUSTOM_EXPONENT_BIT_WIDTH, CUSTOM_MANTISSA_BIT_WIDTH)        (((1 << (CUSTOM_EXPONENT_BIT_WIDTH)) - 1) & ((x) >> (CUSTOM_MANTISSA_BIT_WIDTH)))
#define CUSTOM_GET_MANTISSA(x, CUSTOM_EXPONENT_BIT_WIDTH, CUSTOM_MANTISSA_BIT_WIDTH)        (((1 << (CUSTOM_MANTISSA_BIT_WIDTH)) | (((1 << (CUSTOM_MANTISSA_BIT_WIDTH)) - 1) & (x))) << (23 - (CUSTOM_MANTISSA_BIT_WIDTH)))
#define CUSTOM_GET_EXPONENT_SIGN(x, CUSTOM_EXPONENT_BIT_WIDTH, CUSTOM_MANTISSA_BIT_WIDTH)   (CUSTOM_GET_EXPONENT(x, CUSTOM_EXPONENT_BIT_WIDTH, CUSTOM_MANTISSA_BIT_WIDTH) & (1 << ((CUSTOM_EXPONENT_BIT_WIDTH) - 1)))

#define BUILD_CUSTOM(s, exponent, mantissa, CUSTOM_EXPONENT_BIT_WIDTH, CUSTOM_MANTISSA_BIT_WIDTH) (((s!=0) << (CUSTOM_EXPONENT_BIT_WIDTH + CUSTOM_MANTISSA_BIT_WIDTH)) | ((((1 << CUSTOM_EXPONENT_BIT_WIDTH) - 1) & (exponent)) << CUSTOM_MANTISSA_BIT_WIDTH) | (((mantissa) >> (23 - CUSTOM_MANTISSA_BIT_WIDTH)) & ((1 << CUSTOM_MANTISSA_BIT_WIDTH) - 1)))

#define BUILD_FLOAT(s, exponent, mantissa) ((0x80000000 & ((s) << 31)) | (0x7f800000 & (((exponent) + 0x7f) << 23)) | ((mantissa) & 0x7FFFFF))

#define CORRECTION false

float * TensorQuantizer_quantize (const float * tensor, size_t size)
{
  if (type_array[tensor_id] != target_type)
  {
    SignFormat         sign;
    ExponentFormat     exponent;
    MagnitudeFormat    mantissa;

    SignFormat        f_s;
    ExponentFormat    f_e;
    ExponentFormat    f_es;
    MagnitudeFormat   f_m;


    uint32_t           data;
    float              float_recovered;
    uint32_t           custom_value;
    TensorQuantization * quant_target = nullptr;

    for (size_t i = 0; i < sizeof(quantizations)/sizeof(TensorQuantization); i ++)
    {
      if (quantizations[i].type == target_type)
      {
        quant_target = &quantizations[i];
        break;
      }
    }

    for (size_t i = 0; i < size; i++)
    {
      data = *(uint32_t*) &tensor[i];
      sign = DATA32_GET_SIGN(data);
      exponent = DATA32_GET_EXPONENT(data);
      mantissa = DATA32_GET_MANTISSA(data);

#if CORRECTION
      mantissa &= 0x7FFFFF;
      if ((0x400000 >> quant_target->mantissa) < ((0x7FFFFF >> quant_target->mantissa) & mantissa))
      {
        mantissa = mantissa + (0x400000 >> quant_target->mantissa);
      }

      if (0x800000 & mantissa)
      {
        exponent++;
      }
#endif

      if (exponent < - ((1 << (quant_target->exponent - quant_target->sign)) - 1))
      {
        custom_value = 0;
      }
      else if (exponent > ((1 << (quant_target->exponent - quant_target->sign)) - 1))
      {
        custom_value = BUILD_CUSTOM(sign,
                                   ((1 << (quant_target->exponent - quant_target->sign)) - 1),
                                   0xFFFFFFFF,
                                   quant_target->exponent,
                                   quant_target->mantissa);
      }
      else
      {
        custom_value = BUILD_CUSTOM(sign,
                                   exponent,
                                   mantissa,
                                   quant_target->exponent,
                                   quant_target->mantissa);
      }

      f_s = CUSTOM_GET_SIGN(custom_value,
                            quant_target->exponent,
                            quant_target->mantissa);

      f_e = CUSTOM_GET_EXPONENT(custom_value,
                                quant_target->exponent,
                                quant_target->mantissa);

      f_es = CUSTOM_GET_EXPONENT_SIGN(custom_value,
                                   quant_target->exponent,
                                   quant_target->mantissa);
      if (f_es)
      {
        f_e = ((-1)&~(f_es - 1)) | f_e;
      }

      f_m = CUSTOM_GET_MANTISSA(custom_value,
                                quant_target->exponent,
                                quant_target->mantissa);

      (*(uint32_t*) (&float_recovered)) = BUILD_FLOAT(f_s, f_e, f_m);

      tensors[tensor_id][i] = float_recovered;
    }
    type_array[tensor_id] = target_type;
  }

  return tensors[tensor_id];
}

const char * TensorQuantizer_getTypeName (int id)
{
  return quantization_name[id];
}

}
