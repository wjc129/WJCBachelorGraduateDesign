#include "bp.h"
#include "helper_math.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
__global__ void bproj(float* VolumeBackProj_device, float* residual_device, int* volumeSize_device, int* projSize_device, float* projVector_device, float volbiasz, float dSampleInterval, float dSliceInterval, int anglenums, cudaTextureObject_t my_tex, int angle) {
	uint3 volumeIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (volumeIdx.x > volumeSize_device[0] || volumeIdx.y > volumeSize_device[1]) {
		return;
	}
	float3 sourcePosition = make_float3(projVector_device[angle * 12], projVector_device[angle * 12 + 1], projVector_device[angle * 12 + 2]);
	float3 detectorPosition = make_float3(projVector_device[angle * 12 + 3], projVector_device[angle * 12 + 4], projVector_device[angle * 12 + 5]);
	float3 u = make_float3(projVector_device[angle * 12 + 6], projVector_device[angle * 12 + 7], projVector_device[angle * 12 + 8]);
	float3 v = make_float3(projVector_device[angle * 12 + 9], projVector_device[angle * 12 + 10], projVector_device[angle * 12 + 11]);
	float3 volumeCenter = make_float3(volumeSize_device[0] / -2.0f, volumeSize_device[1] / -2.0f, volumeSize_device[2] / -2.0f);
	float2 detectorCenter = make_float2(projSize_device[0] / -2.0f, projSize_device[1] / -2.0f);
	for (int z0 = 0; z0 < volumeSize_device[2]; z0++) {
		int volume_id = z0 * volumeSize_device[1] * volumeSize_device[0] + volumeIdx.y * volumeSize_device[0] + volumeIdx.x;
		float3 coordinates = make_float3((volumeCenter.x + volumeIdx.x)* dSampleInterval, (volumeCenter.y + volumeIdx.y)* dSampleInterval, (volumeCenter.z + z0)* dSliceInterval + volbiasz);
		float grid[3][3];
		grid[0][0] = u.x; grid[1][0] = u.y; grid[2][0] = u.z;
		grid[0][1] = v.x; grid[1][1] = v.y; grid[2][1] = v.z;
		grid[0][2] = (sourcePosition.x - coordinates.x);
		grid[1][2] = (sourcePosition.y - coordinates.y);
		grid[2][2] = (sourcePosition.z - coordinates.z);
		
		float Scale_ = grid[0][0] * (grid[1][1] * grid[2][2] - grid[1][2] * grid[2][1])
			- grid[0][1] * (grid[1][0] * grid[2][2] - grid[1][2] * grid[2][0])
			+ grid[0][2] * (grid[1][0] * grid[2][1] - grid[1][1] * grid[2][0]);;
		if (Scale_ == 0);
		else {
			Scale_ = 1.0f / Scale_;
		}
		grid[0][0] = (coordinates.x - sourcePosition.x);
		grid[1][0] = (coordinates.y - sourcePosition.y);
		grid[2][0] = (coordinates.z - sourcePosition.z);
		grid[0][1] = v.x; grid[1][1] = v.y; grid[2][1] = v.z;
		grid[0][2] = (sourcePosition.x - detectorPosition.x);
		grid[1][2] = (sourcePosition.y - detectorPosition.y);
		grid[2][2] = (sourcePosition.z - detectorPosition.z);
		float det_Scale_1 = grid[0][0] * (grid[1][1] * grid[2][2] - grid[1][2] * grid[2][1])
			- grid[0][1] * (grid[1][0] * grid[2][2] - grid[1][2] * grid[2][0])
			+ grid[0][2] * (grid[1][0] * grid[2][1] - grid[1][1] * grid[2][0]);
		//det3(u, coordinates - sourcePosition, sourcePosition - detectorPosition) 
		
		grid[0][0] = u.x; grid[1][0] = u.y; grid[2][0] = u.z;
		grid[0][1] = coordinates.x - sourcePosition.x;
		grid[1][1] = coordinates.y - sourcePosition.y;
		grid[2][1] = coordinates.z - sourcePosition.z;
		grid[0][2] = sourcePosition.x - detectorPosition.x;
		grid[1][2] = sourcePosition.y - detectorPosition.y;
		grid[2][2] = sourcePosition.z - detectorPosition.z;
		float det_Scale_2 = grid[0][0] * (grid[1][1] * grid[2][2] - grid[1][2] * grid[2][1])
			- grid[0][1] * (grid[1][0] * grid[2][2] - grid[1][2] * grid[2][0])
			+ grid[0][2] * (grid[1][0] * grid[2][1] - grid[1][1] * grid[2][0]);


		//detectorX,detectorY
		float detectorX = Scale_ * det_Scale_1 - detectorCenter.x;
		float detectorY = Scale_ * det_Scale_2 - detectorCenter.y;
		VolumeBackProj_device[volume_id] += (tex3D<float>(my_tex, detectorX, detectorY, angle + 0.5f) / anglenums);

	}
	
//}
}
//const dim3 blockSize_vol = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);
//const dim3 gridSize_vol = dim3(volumeWidth / BLOCK_X + 1, volumeDepth / BLOCK_Y + 1, volumeHeight / BLOCK_Z + 1);
__global__ void linetrans(float* volume_device, float* VolumeBackProj_device, int* volumeSize_device,float miu) {
	uint3 volumeIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (volumeIdx.x > volumeSize_device[0] || volumeIdx.y > volumeSize_device[1]) {
		return;
	}
	for(int z0=0;z0< volumeSize_device[2];z0++){
		int volume_id = z0 * volumeSize_device[1] * volumeSize_device[0] + volumeIdx.y * volumeSize_device[0] + volumeIdx.x;
		//volume = volume - miu * result;
		volume_device[volume_id] = volume_device[volume_id] - miu * VolumeBackProj_device[volume_id];
		VolumeBackProj_device[volume_id] = 0;
	}
}

__global__ void relutrans(float* volume_device, int* volumeSize_device) {
	uint3 volumeIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (volumeIdx.x > volumeSize_device[0] || volumeIdx.y > volumeSize_device[1]) {
		return;
	}
	for (int z0 = 0; z0 < volumeSize_device[2]; z0++) {
		int volume_id = z0 * volumeSize_device[1] * volumeSize_device[0] + volumeIdx.y * volumeSize_device[0] + volumeIdx.x;
		if (volume_device[volume_id] < 0) {
			volume_device[volume_id] = 0;
		}
	}
}

