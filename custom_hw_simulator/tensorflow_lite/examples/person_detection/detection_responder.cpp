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

#include "detection_responder.h"
#include "model_settings.h"

#include "stdlib.h"
#include "string.h"
#include "stdio.h"

static float accuracy = 0;
static int num_samples = 0;
static int correct_samples = 0;

// This dummy implementation writes person and no person scores to the error
// console. Real applications will want to take some custom action instead, and
// should implement their own versions of this function.
void RespondToDetection (tflite::ErrorReporter* error_reporter,
                         TfLiteTensor* output,
                         int expected_index,
                         int verbose)
{
  char message[80] = { 0 };
  float temp;
  int index[3] = { 0 };

  temp = 0;
  for (int i = 0; i < output->dims->data[1]; i++)
  {
    if (temp < output->data.f[i])
    {
      temp = output->data.f[i];
      index[2] = index[1];
      index[1] = index[0];
      index[0] = i;
    }
  }

  num_samples ++;
  if (expected_index == index[0])
  {
    correct_samples += expected_index == index[0];
    if (1 < verbose)
    {
      printf ("%f [%s]\n",
              output->data.f[index[0]],
              CifarClassLabels[index[0]]);
    }
  }
  else
  {
    if (1 < verbose)
    {
      printf ("[FAIL]: %f [%s], expected: [%s]\n",
              output->data.f[index[0]],
              CifarClassLabels[index[0]],
              CifarClassLabels[expected_index]);
    }
  }

  accuracy = ((float) correct_samples) / ((float) num_samples);

  if (0 < verbose)
  {
    printf ("%d, Acc: %.4f\n\n", num_samples, accuracy);
  }
}

void ResetStatistics ()
{
  accuracy = 0;
  num_samples = 0;
  correct_samples = 0;
}

void GetAccuracy (float * accuracy_ptr, int * num_samples_ptr, int * correct_samples_ptr)
{
  if (accuracy_ptr)
  {
    *accuracy_ptr =  accuracy;
  }

  if (num_samples_ptr)
  {
    *num_samples_ptr =  num_samples;
  }

  if (correct_samples_ptr)
  {
    *correct_samples_ptr =  correct_samples;
  }
}
