/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "TensorFlowLite.h"

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "custom_quantizer.h"

//#include "quantizer.h"
////////////////////////////////////////////////////////////////////
// Xilinx libraries
#include <iostream>
#include <fstream>


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 256 * 1024 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

static int File_readData (const char * file_name, void * model)
{
  std::ifstream file (file_name, std::ios::in|std::ios::binary|std::ios::ate);
  int rc = file.is_open();

  assert(rc);
  if (rc)
  {
    size_t size = file.tellg();
    file.seekg (0, std::ios::beg);
    file.read ((char*)model, size);

    file.close();
  }

  return rc;
}

//static int File_writeData (const char * file_name, const void * data, size_t size)
//{
//  std::ifstream file (file_name, std::ios::out|std::ios::binary|std::ios::ate);
//  int rc = file.is_open();
//
//  assert(rc);
//  if (rc)
//  {
//    file.seekg (0, std::ios::beg);
//    file.write ((char*)data, size);
//
//    file.close();
//  }
//
//  return rc;
//}

typedef struct
{
  custom_float::CustomFloatType quantization_type;
  float                         accuracy;
} AccuracyReport;

typedef struct
{
  const char *   model_name;
  AccuracyReport accuracy_report[16];
  int            accuracy_report_size;
  int            samples;
} ModelReport;


ModelReport model_report[] =
{
  {
    .model_name = "models/vgg_f32",
    .accuracy_report = {
      {
        .quantization_type = custom_float::CUSTOM_150,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::CUSTOM_151,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::CUSTOM_152,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::CUSTOM_153,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::CUSTOM_154,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::FP16,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::BFLOAT16,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::TENSOR_FLOAT,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::FP32,
        .accuracy = 0
      }
    },
    .accuracy_report_size = 9,
    .samples = 10
  },
  {
    .model_name = "models/mob_f32",
    .accuracy_report = {
      {
        .quantization_type = custom_float::CUSTOM_150,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::CUSTOM_151,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::CUSTOM_152,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::CUSTOM_153,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::CUSTOM_154,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::FP16,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::BFLOAT16,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::TENSOR_FLOAT,
        .accuracy = 0
      },
      {
        .quantization_type = custom_float::FP32,
        .accuracy = 0
      }
    },
    .accuracy_report_size = 9,
    .samples = 10
  }
};

static const int model_index = 0;


void print_report (void)
{
  printf ("\nModel name: %s\n", model_report[model_index].model_name);
  printf ("Number of samples: %d\n", model_report[model_index].samples);

  for (int i = 0; i < model_report[model_index].accuracy_report_size; i ++)
  {
    printf ("%s = %f\n",
            TensorQuantizer_getTypeName(model_report[model_index].accuracy_report[i].quantization_type),
            model_report[model_index].accuracy_report[i].accuracy);
  }
}


unsigned char model_data[4966272];

unsigned char labels[10000];

// The name of this function is important for Arduino compatibility.


static int image_index = 0;
static int quantization_index = 0;

void setup ()
{
  int rc;

  tflite::InitializeTarget ();

//  rc = File_readData ("models/mob_f32", model_data);

  rc = File_readData (model_report[model_index].model_name, model_data);

  assert(rc == 1);

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel (model_data);

  if (model->version () != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version (), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver micro_op_resolver;

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter (model, micro_op_resolver,
                                                      tensor_arena,
                                                      kTensorArenaSize,
                                                      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors ();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input (0);

  TF_LITE_REPORT_ERROR(error_reporter, "input->dims->size = %d",
                       input->dims->size);
  for (int i = 0; i < input->dims->size; i++)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "input->dims->data[%d] = %d", i,
                         input->dims->data[i]);
  }
  TF_LITE_REPORT_ERROR(error_reporter, "input->type = 0x%d", input->type);

//////////////////////////////////////////////////////////////////////////////////
  TfLiteTensor* output = interpreter->output (0);

  TF_LITE_REPORT_ERROR(error_reporter, "output->dims->size = %d",
                       output->dims->size);
  for (int i = 0; i < output->dims->size; i++)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "output->dims->data[%d] = %d", i,
                         output->dims->data[i]);
  }
  TF_LITE_REPORT_ERROR(error_reporter, "input->type = 0x%d", output->type);

  rc = File_readData ("CIFAR/labels", labels);

  ResetStatistics();
  image_index = 0;
  quantization_index = 0;
}

// The name of this function is important for Arduino compatibility.
void loop ()
{
  char img_name[32] = { 0 };
  int rc;

  TfLiteStatus status;

  if (image_index == 0)
  {
    custom_float::CustomFloatType quantization_type = model_report[model_index].accuracy_report[quantization_index].quantization_type;
    printf ("\nStarting %s\n", TensorQuantizer_getTypeName(quantization_type));
    custom_float::TensorQuantizer_setType (quantization_type);
  }


  sprintf(img_name, "CIFAR/%d", image_index);

  rc = File_readData (img_name, input->data.data);
  assert(rc == 1);

  if (rc != 1)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }

  status = interpreter->Invoke ();

  if (status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output (0);

  RespondToDetection (error_reporter, output, labels[image_index], 1);

  image_index++;

  if (model_report[model_index].samples <= image_index)
  {
    GetAccuracy (&model_report[model_index].accuracy_report[quantization_index].accuracy, nullptr, nullptr);
    ResetStatistics ();
    image_index = 0;

    printf ("\nEnding %s\n", TensorQuantizer_getTypeName(model_report[model_index].accuracy_report[quantization_index].quantization_type));
    quantization_index ++;

    if (model_report[model_index].accuracy_report_size <= quantization_index)
    {
      printf ("\nDone!\n");

      print_report ();

      while (1);
    }
  }
}
