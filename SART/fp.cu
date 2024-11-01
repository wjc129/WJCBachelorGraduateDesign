


#include <stdio.h>

#include "fp.h"
#include<math.h>



__global__ void fproj(float* sinogram_real_device, float* residual_device, float* projVector_device, int* volumeSize_device, int* projSize_device, float volbiasz, float dSampleInterval, float dSliceInterval, int anglenums,
	cudaTextureObject_t my_tex, int angle) {
	uint3 detectorIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (detectorIdx.x > projSize_device[0] || detectorIdx.y > projSize_device[1]) {
		return;
	}
	//float2 detectorCenter = make_float2(detectorSize) / -2.0;
	float2 detectorCenter = make_float2(projSize_device[0] / -2.0, projSize_device[1] / -2.0);
	float3 volumeCenter = make_float3(volumeSize_device[0] / -2.0, volumeSize_device[1] / -2.0, volumeSize_device[2] / -2.0);

	// 获取探测器坐标(包含负值的实际坐标)
	float detectorX = detectorIdx.x + detectorCenter.x;
	float detectorY = detectorIdx.y + detectorCenter.y;
	//for (int angle = 0; angle < anglenums; angle++) {
	int id_proj = angle * projSize_device[0] * projSize_device[1] + detectorIdx.y * projSize_device[0] + detectorIdx.x;
	float3 sourcePosition = make_float3(projVector_device[angle * 12], projVector_device[angle * 12 + 1], projVector_device[angle * 12 + 2]);
	float3 detectorPosition = make_float3(projVector_device[angle * 12 + 3], projVector_device[angle * 12 + 4], projVector_device[angle * 12 + 5]);
	float3 u = make_float3(projVector_device[angle * 12 + 6], projVector_device[angle * 12 + 7], projVector_device[angle * 12 + 8]);
	float3 v = make_float3(projVector_device[angle * 12 + 9], projVector_device[angle * 12 + 10], projVector_device[angle * 12 + 11]);

	float3 detectorPixel;
	detectorPixel.x = detectorPosition.x + (0.5f + detectorX) * u.x + (0.5f + detectorY) * v.x;
	detectorPixel.y = detectorPosition.y + (0.5f + detectorX) * u.y + (0.5f + detectorY) * v.y;
	detectorPixel.z = detectorPosition.z + (0.5f + detectorX) * u.z + (0.5f + detectorY) * v.z;
	float3 rayline;
	rayline.x = detectorPixel.x - sourcePosition.x;
	rayline.y = detectorPixel.y - sourcePosition.y;
	rayline.z = detectorPixel.z - sourcePosition.z;
	float len_ray = sqrtf(rayline.x * rayline.x + rayline.y * rayline.y + rayline.z * rayline.z);
	float3 rayVector;
	rayVector.x = rayline.x / len_ray;
	rayVector.y = rayline.y / len_ray;
	rayVector.z = rayline.z / len_ray;
	float pixel = 0.0f;
	float alpha0, alpha1;
	float rayVectorDomainDim = fmax(fabs(rayVector.x), fmax(fabs(rayVector.z), fabs(rayVector.y)));
	if (fabs(rayVector.x) == rayVectorDomainDim) {
		float volume_min_edge_point = volumeCenter.x * dSampleInterval;
		float volume_max_edge_point = (volumeSize_device[0] + volumeCenter.x) * dSampleInterval;
		alpha0 = (volume_min_edge_point - sourcePosition.x) / rayVector.x;
		alpha1 = (volume_max_edge_point - sourcePosition.x) / rayVector.x;
	}
	else if (fabs(rayVector.y) == rayVectorDomainDim) {
		float volume_min_edge_point = volumeCenter.y * dSampleInterval;
		float volume_max_edge_point = (volumeSize_device[1] + volumeCenter.y) * dSampleInterval;
		alpha0 = (volume_min_edge_point - sourcePosition.y) / rayVector.y;
		alpha1 = (volume_max_edge_point - sourcePosition.y) / rayVector.y;
	}
	else {
		float volume_min_edge_point = volumeCenter.z * dSliceInterval + volbiasz;
		float volume_max_edge_point = (volumeSize_device[2] + volumeCenter.z) * dSliceInterval + volbiasz;
		alpha0 = (volume_min_edge_point - sourcePosition.z) / rayVector.z;
		alpha1 = (volume_max_edge_point - sourcePosition.z) / rayVector.z;
	}

	float min_alpha = fmin(alpha0, alpha1) - 3;
	float max_alpha = fmax(alpha0, alpha1) + 3;
	float px = (sourcePosition.x + min_alpha * rayVector.x) / dSampleInterval - volumeCenter.x;
	float py = (sourcePosition.y + min_alpha * rayVector.y) / dSampleInterval - volumeCenter.y;
	float pz = (sourcePosition.z + min_alpha * rayVector.z - volbiasz) / dSampleInterval - volumeCenter.z;
	float step_size = 1;
	float len = max_alpha - min_alpha;
	while (min_alpha < max_alpha)
	{
		pixel += tex3D<float>(my_tex, px + 0.5f, py + 0.5f, pz + 0.5f);
		px += rayVector.x; py += rayVector.y; pz += rayVector.z; min_alpha++;
	}
	pixel *= step_size;
	residual_device[id_proj] = (pixel - sinogram_real_device[id_proj]) / len;
}
