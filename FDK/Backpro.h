#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
__global__ void back_projection(float* shepplogan3D_reconstruct_device, float* projVector_device, int* volumeSize_device, int* projSize_device, float volbiasz, float dSampleInterval, float dSliceInterval,
    cudaTextureObject_t my_tex, int angle, int anglenums);