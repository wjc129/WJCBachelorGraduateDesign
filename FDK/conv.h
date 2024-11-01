#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>

__global__ void Conv_Kernal(float* F_weight_ori, int size_F, float* filter_g, int size_filter, float* result_ori, int anglenums
	, int projWidth, int projHeight);