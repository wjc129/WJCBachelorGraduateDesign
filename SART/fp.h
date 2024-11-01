#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void fproj(float* sinogram_real_device, float* residual_device, float* projVector_device, int* volumeSize_device, int* projSize_device, float volbiasz, float dSampleInterval, float dSliceInterval, int anglenums,
	cudaTextureObject_t my_tex, int angle);