#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void bproj(float* VolumeBackProj_device, float* residual_device, int* volumeSize_device, int* projSize_device, float* projVector_device, float volbiasz, float dSampleInterval, float dSliceInterval, int anglenums, cudaTextureObject_t my_tex, int angle);

__global__ void linetrans(float* volume_device, float* VolumeBackProj_device, int* volumeSize_device, float miu);

__global__ void relutrans(float* volume_device, int* volumeSize_device);



