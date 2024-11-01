#include "conv.h"

__global__ void Conv_Kernal(float* F_weight_ori, int size_F, float* filter_g, int size_filter, float* result_ori, int anglenums
    , int projWidth, int projHeight) {

    uint3 projId = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
    if (projId.x > anglenums || projId.y > size_F) {
        return;
    }
    int beltaid = projId.x;
    int z = projId.y;
    bool sign = 1;
    int offset = 0;
    //判断序列中心位置是否为整数，若为整数，sign赋值为0
    if ((size_F + size_filter - 2) % 2 == 0)
        sign = 0;
    offset = (size_F + size_filter - 2) / 2 - ((size_F - 1) / 2) + sign;
    int max_length = anglenums * projHeight * projWidth;
    int bias = beltaid * projWidth * projHeight + z * projWidth;
    for (int n = 0; n < size_F + size_filter - 1; ++n) {
        float mid = 0;
        for (int k = 0; k < size_F; ++k) {
            if (n - k >= 0 && n - k < size_filter && k + bias < max_length) {
                mid += F_weight_ori[k + bias] * filter_g[n - k];
            }
        }
        if (n - offset + bias >= 0 && n - offset + bias < max_length)
            result_ori[n - offset + bias] = mid;
    }

}