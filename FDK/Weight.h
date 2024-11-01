#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#define pi 3.14159
__global__ void WeightKernal(float* sinogram_device, int* projSize_device, int* volumeSize_device, float* projVector_device, int angle, int anglenums,
    float dSampleInterval, float dSliceInterval, float volbiasz);
