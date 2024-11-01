#include "Backpro.h"

#define pi  3.1415926
/*
__global__ void back_projection(float* shepplogan3D_reconstruct_device, float* projVector_device, int* volumeSize_device, int* projSize_device, float volbiasz, float dSampleInterval, float dSliceInterval,
    cudaTextureObject_t my_tex, int angle, int anglenums)
{
    uint3 volumeId = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
    if (volumeId.x > volumeSize_device[0] || volumeId.y > volumeSize_device[1]) {
        return;
    }
    //投影几何各个变量数据进行初始化
    float3 sourcePosition = make_float3(projVector_device[angle * 12], projVector_device[angle * 12 + 1], projVector_device[angle * 12 + 2]);
    float3 detectorPosition = make_float3(projVector_device[angle * 12 + 3], projVector_device[angle * 12 + 4], projVector_device[angle * 12 + 5]);
    float3 volumeCenter = make_float3(volumeSize_device[0] / -2.0f, volumeSize_device[1] / -2.0f, volumeSize_device[2] / -2.0f);
    float2 detectorCenter = make_float2(projSize_device[0] / -2.0f, projSize_device[1] / -2.0f);
    int volume_x = blockIdx.x * blockDim.x + threadIdx.x;
    int volume_y = blockIdx.y * blockDim.y + threadIdx.y;
    int volumeWidth = volumeSize_device[0];
    int volumeDepth = volumeSize_device[1];
    int volumeHeight = volumeSize_device[2];
    float DOD = sqrtf(detectorPosition.x * detectorPosition.x + detectorPosition.y * detectorPosition.y +
        detectorPosition.z * detectorPosition.z);
    float D = sqrtf(sourcePosition.x * sourcePosition.x + sourcePosition.y * sourcePosition.y +
        sourcePosition.z * sourcePosition.z);
    for (int z0 = 0; z0 < volumeHeight; z0++) {
        float t = (volume_x + volumeCenter.x) * dSampleInterval * (sourcePosition.x / D) + (volume_y + volumeCenter.y) * dSampleInterval * (sourcePosition.y / D);
        float s = -(volume_x + volumeCenter.x) * dSampleInterval * (sourcePosition.y / D) + (volume_y + volumeCenter.y) * dSampleInterval * (sourcePosition.x / D);
        float t_proj = (D + DOD) * t / (D - s);
        float z_proj = (D + DOD) * (z0 + volumeCenter.z) * dSliceInterval / (D - s);
        float mid = (D / (D - s)) * (D / (D - s));
        float per_angle = 2 * pi / anglenums;
        int num = volume_x + volume_y * volumeWidth + z0 * volumeWidth * volumeDepth;
        int length = volumeWidth * volumeDepth * volumeHeight;
        if (num >= 0 && num < length) {
            //shepplogan3D_reconstruct_device[num] = tex3D<float>(my_tex, z_proj - detectorCenter.y + 0.5f, t_proj - detectorCenter.x + 0.5f, angle + 0.5f) * mid * per_angle / 2;
            shepplogan3D_reconstruct_device[num] += tex3D<float>(my_tex, t_proj - detectorCenter.x + 0.5f, z_proj - detectorCenter.y + 0.5f, angle + 0.5f) * mid * per_angle / 2;
        }
    }

}
*/
__global__ void back_projection(float* shepplogan3D_reconstruct_device, float* projVector_device, int* volumeSize_device, int* projSize_device, float volbiasz, float dSampleInterval, float dSliceInterval,
    cudaTextureObject_t my_tex, int angle, int anglenums)
{
    uint3 volumeId = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
    if (volumeId.x > volumeSize_device[0] || volumeId.y > volumeSize_device[1]) {
        return;
    }
    //投影几何各个变量数据进行初始化
    float3 sourcePosition = make_float3(projVector_device[angle * 12], projVector_device[angle * 12 + 1], projVector_device[angle * 12 + 2]);
    float3 detectorPosition = make_float3(projVector_device[angle * 12 + 3], projVector_device[angle * 12 + 4], projVector_device[angle * 12 + 5]);
    float3 volumeCenter = make_float3(volumeSize_device[0] / -2.0f, volumeSize_device[1] / -2.0f, volumeSize_device[2] / -2.0f);
    float2 detectorCenter = make_float2(projSize_device[0] / -2.0f, projSize_device[1] / -2.0f);
    float3 u = make_float3(projVector_device[angle * 12 + 6], projVector_device[angle * 12 + 7], projVector_device[angle * 12 + 8]);
    float3 v = make_float3(projVector_device[angle * 12 + 9], projVector_device[angle * 12 + 10], projVector_device[angle * 12 + 11]);
    int volume_x = blockIdx.x * blockDim.x + threadIdx.x;
    int volume_y = blockIdx.y * blockDim.y + threadIdx.y;
    int volumeWidth = volumeSize_device[0];
    int volumeDepth = volumeSize_device[1];
    int volumeHeight = volumeSize_device[2];


    // 计算由u和v组成的平面的法向量
    float3 orient = make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
    float orient_leng = sqrtf(orient.x * orient.x + orient.y * orient.y + orient.z * orient.z);
    float3 per_orient = make_float3(orient.x / orient_leng, orient.y / orient_leng, orient.z / orient_leng);
    float3 s_d_vector = make_float3(detectorPosition.x - sourcePosition.x, detectorPosition.y - sourcePosition.y, detectorPosition.z - sourcePosition.z);
    float D_temp = per_orient.x * s_d_vector.x + per_orient.y * s_d_vector.y + per_orient.z * s_d_vector.z;
    float D;
    if (D_temp < 0) {
        D = -D_temp;
        per_orient = make_float3(-per_orient.x, -per_orient.y, -per_orient.z);
    }
    else {
        D = D_temp;
    }
    float3 O_cordinate = make_float3(sourcePosition.x + D * per_orient.x, sourcePosition.y + D * per_orient.y, sourcePosition.z + D * per_orient.z);
    float3 O_dc_vector = make_float3(detectorPosition.x - O_cordinate.x, detectorPosition.y - O_cordinate.y, detectorPosition.z - O_cordinate.z);
    for (int volume_z = 0; volume_z < volumeHeight; volume_z++) {
        // get the true x,y 
        float x_true = (volume_x + volumeCenter.x) * dSampleInterval;
        float y_true = (volume_y + volumeCenter.y) * dSampleInterval;
        float z_true = (volume_z + volumeCenter.z) * dSliceInterval + volbiasz;
        // change to source detector cordinate
        // get the s_per_vector
        float3 s_per_vector = make_float3(-per_orient.x, -per_orient.y, -per_orient.z);
        // get the t_per_vector
        float3 t_per_vector = make_float3(u.x, u.y, u.z);
        float3 z_per_vector = make_float3(v.x, v.y, v.z);
        //get the true s,t out of the new s-t
        float3 O_Point = make_float3(x_true - O_cordinate.x, y_true - O_cordinate.y, z_true - O_cordinate.z);
        float O_Point_s = O_Point.x * s_per_vector.x + O_Point.y * s_per_vector.y + O_Point.z * s_per_vector.z;
        float O_Point_t = O_Point.x * t_per_vector.x + O_Point.y * t_per_vector.y + O_Point.z * t_per_vector.z;
        float O_Point_z = O_Point.x * z_per_vector.x + O_Point.y * z_per_vector.y + O_Point.z * z_per_vector.z;
        
        // get the bias between o and center
        float O_dc_vector_t = O_dc_vector.x * t_per_vector.x + O_dc_vector.y * t_per_vector.y + O_dc_vector.z * t_per_vector.z;
        float O_dc_vector_z = O_dc_vector.x * z_per_vector.x + O_dc_vector.y * z_per_vector.y + O_dc_vector.z * z_per_vector.z;
        float t_proj = D * O_Point_t / (D - O_Point_s);
        float z_proj = D * O_Point_z / (D - O_Point_s);
        float mid = (D / (D - O_Point_s)) * (D / (D - O_Point_s));
        float per_angle = 2 * pi / anglenums;
        int num = volume_x + volume_y * volumeWidth + volume_z * volumeWidth * volumeDepth;
        int length = volumeWidth *volumeDepth * volumeHeight;
        if (num >= 0 && num < length) {
            shepplogan3D_reconstruct_device[num]+= tex3D<float>(my_tex, t_proj - detectorCenter.x- O_dc_vector_t +0.5f, z_proj - detectorCenter.y - O_dc_vector_z +0.5f, angle + 0.5f) * mid * per_angle / 2;
            
            //shepplogan3D_reconstruct_device[num] += tex3D<float>(my_tex, t_proj - detectorCenter.x + 0.5f, z_proj - detectorCenter.y + 0.5f, angle + 0.5f) * mid * per_angle / 2;
        }
    }

}
