#include "Weight.h"


__global__ void WeightKernal(float* sinogram_device,int* projSize_device,int* volumeSize_device,float* projVector_device,int angle,int anglenums,
    float dSampleInterval,float dSliceInterval,float volbiasz)
{
	uint3 projId = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (projId.x > projSize_device[0] || projId.y > projSize_device[1]) {
		return;
	}
    int volume_x = blockIdx.x * blockDim.x + threadIdx.x;
    int volume_y = blockIdx.y * blockDim.y + threadIdx.y;
    int volumeWidth = volumeSize_device[0];
    int volumeDepth = volumeSize_device[1];
    int volumeHeight = volumeSize_device[2];
    int projWidth = projSize_device[0];
    int projHeight = projSize_device[1];
	float3 sourcePosition = make_float3(projVector_device[angle * 12], projVector_device[angle * 12 + 1], projVector_device[angle * 12 + 2]);
	float3 detectorPosition = make_float3(projVector_device[angle * 12 + 3], projVector_device[angle * 12 + 4], projVector_device[angle * 12 + 5]);
	float3 u = make_float3(projVector_device[angle * 12 + 6], projVector_device[angle * 12 + 7], projVector_device[angle * 12 + 8]);
	float3 v = make_float3(projVector_device[angle * 12 + 9], projVector_device[angle * 12 + 10], projVector_device[angle * 12 + 11]);
	float3 volumeCenter = make_float3(volumeSize_device[0] / -2.0f, volumeSize_device[1] / -2.0f, volumeSize_device[2] / -2.0f);
	float2 detectorCenter = make_float2(projSize_device[0] / -2.0f, projSize_device[1] / -2.0f);
	// 计算由u和v组成的平面的法向量
	float3 orient = make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
	float orient_leng = sqrtf(orient.x * orient.x + orient.y * orient.y + orient.z * orient.z);
	float3 per_orient = make_float3(orient.x/orient_leng,orient.y/orient_leng,orient.z/orient_leng);
	float3 s_d_vector = make_float3(detectorPosition.x- sourcePosition.x, detectorPosition.y- sourcePosition.y, detectorPosition.z- sourcePosition.z);
	float D_temp = per_orient.x * s_d_vector.x + per_orient.y * s_d_vector.y + per_orient.z * s_d_vector.z;
	float D;
	if (D_temp < 0) {
		D = -D_temp;
		per_orient = make_float3(-per_orient.x,-per_orient.y,-per_orient.z);
	}
	else {
		D = D_temp;
	}
	float3 O_cordinate = make_float3(sourcePosition.x+D*per_orient.x,sourcePosition.y+D*per_orient.y,sourcePosition.z+D*per_orient.z);
	float3 O_dc_vector = make_float3(detectorPosition.x-O_cordinate.x,detectorPosition.y-O_cordinate.y,detectorPosition.z-O_cordinate.z);
	float O_dc_vector_t = O_dc_vector.x * u.x + O_dc_vector.y * u.y + O_dc_vector.z * u.z;
	float O_dc_vector_z = O_dc_vector.x * v.x + O_dc_vector.y * v.y + O_dc_vector.z * v.z;
	float x_true = projId.x + detectorCenter.x+ O_dc_vector_t;
	float y_true = projId.y + detectorCenter.y + O_dc_vector_z;
	sinogram_device[angle * projWidth * projHeight + projId.y * projWidth + projId.x] *= (D / sqrtf(D * D + y_true * y_true + x_true * x_true));
    
    

}


