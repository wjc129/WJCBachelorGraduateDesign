//  创建头文件的原因:若只是 #include "fp.cu" 编译会不要cuda而不会cpp
#pragma warning(disable:4996)
#include "fp.h"
#include "bp.h"
#include <cstring>
#include <iostream>
#include <chrono>
#include <string>
#include <time.h>
#define pi 3.14159

using namespace std;


int main()
{
	// 设置射线扫描物体参数
/*
	const int volumeWidth = 600;
	const int volumeHeight = 100;
	const int volumeDepth = 600;
	float volbiasz = -900.0f;
	float dSampleInterval = 0.8f;
	float dSliceInterval = 0.8f;
	// 设置探测器板参数
	const int projWidth =1600;
	const int projHeight = 1600;
	// 设置投影角度总数
	const int anglenums = 32;
	// 设置迭代次数
	const int IterNum =100;
		*/
	const int volumeWidth = 256;
	const int volumeHeight = 256;
	const int volumeDepth = 256;
	float volbiasz = 0.0f;
	float dSampleInterval = 1.0f;
	float dSliceInterval = 1.0f;
	const int projWidth = 799;
	const int projHeight = 799;
	// 设置投影角度总数
	const int anglenums = 36;
	// 设置迭代次数
	const int IterNum = 100;
	// 用一个长度为3的int类型数组存储几何体几何参数
	int* volumeSize = new int[3];
	volumeSize[0] = volumeWidth;
	volumeSize[1] = volumeDepth;
	volumeSize[2] = volumeHeight;
	
	// 用一个长度为2的int类型数组存储探测器几何参数
	int* projSize = new int[2];
	projSize[0] = projWidth;
	projSize[1] = projHeight;
	
	// 设置残差更新初始比率
	float miu =1;
	// 残差比率更新比率参数
	float beta = 0.8;
	// 设置GPU内存块值
	const int BLOCK_X = 32;
	const int BLOCK_Y = 32;
	// 下述代码区段,读取sinogram投影数据
	float* sinogram_real = new float[projWidth * projHeight * anglenums];
	memset(sinogram_real, 0, projWidth * projHeight * anglenums * sizeof(float));
	//const char* filename = "E:/WJC_train/all_360_proj_data/Proj3601mmB30L067.raw";
	const char* filename = "sinogram_file_sparse.raw";
	FILE* fp = nullptr;
	if ((fp = fopen(filename, "rb")) == nullptr) {
		printf("无法打开 %s 文件 ", filename);
		return false;
	}
	fread(sinogram_real, sizeof(float), projHeight * projWidth * anglenums, fp);
	fclose(fp);

	//初始化正投影数据
	float* sinogram_proj = new float[projHeight * projWidth * anglenums];
	memset(sinogram_proj, 0, projWidth * projHeight * anglenums * sizeof(float));

	// 初始化通过反投影生成的增量信息
	float* VolumeBackProj = new float[volumeHeight * volumeWidth * volumeDepth];
	memset(VolumeBackProj, 0, volumeHeight * volumeWidth * volumeDepth * sizeof(float));
	
	// // 下述代码区段,读取projectVector
	// 读取matlab上存储的geometry文件,文件存储了各个角度的投影数据
	float* projVector = new float[12 * anglenums];
	memset(projVector, 0, 12 * anglenums * sizeof(float));
	//const char* filename_geo = "proj_vec_ss.raw";
	const char* filename_geo = "geometry_sparse.raw";
	FILE* fp2 = nullptr;
	if ((fp2 = fopen(filename_geo, "rb")) == nullptr) {
		printf("无法打开 %s 文件 ", filename_geo);
		return false;
	}

	fread(projVector, sizeof(float), 12 * anglenums, fp2);
	fclose(fp2);

	//  初始化几何体
	float* volume = new float[volumeWidth * volumeHeight * volumeDepth];
	memset(volume, 0, volumeWidth * volumeHeight * volumeDepth * sizeof(float));

	// 初始化残差变量数组
	float* residual = new float[projHeight * projWidth * anglenums];
	memset(residual, 0, projHeight * projWidth * anglenums * sizeof(float));

	//  在设备上也创建数组,并将数组的元素迁移到设备上
	float* volume_device;
	cudaMalloc((void**)&volume_device, volumeWidth * volumeDepth * volumeHeight * sizeof(float));
	cudaMemcpy(volume_device, volume, volumeWidth * volumeDepth * volumeHeight * sizeof(float), cudaMemcpyHostToDevice);
	cudaFree(volume);


	float* sinogram_real_device;
	cudaMalloc((void**)&sinogram_real_device, projHeight * projWidth * anglenums * sizeof(float));
	cudaMemcpy(sinogram_real_device, sinogram_real, projHeight * projWidth * anglenums * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaFree(sinogram_real);
	float* sinogram_proj_device;
	cudaMalloc((void**)&sinogram_proj_device, projHeight * projWidth * anglenums * sizeof(float));
	cudaMemcpy(sinogram_proj_device, sinogram_proj, projHeight * projWidth * anglenums * sizeof(float), cudaMemcpyHostToDevice);
	

	float* residual_device;
	cudaMalloc((void**)&residual_device, projHeight * projWidth * anglenums * sizeof(float));
	cudaMemcpy(residual_device, residual, projHeight * projWidth * anglenums * sizeof(float), cudaMemcpyHostToDevice);
	cudaFree(residual);

	int* volumeSize_device;
	cudaMalloc((void**)&volumeSize_device, 3 * sizeof(int));
	cudaMemcpy(volumeSize_device, volumeSize, 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaFree(volumeSize);

	int* projSize_device;
	cudaMalloc((void**)&projSize_device, 2 * sizeof(int));
	cudaMemcpy(projSize_device, projSize, 2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaFree(projSize);

	float* projVector_device;
	cudaMalloc((void**)&projVector_device, 12 * anglenums * sizeof(float));
	cudaMemcpy(projVector_device, projVector, 12 * anglenums * sizeof(float), cudaMemcpyHostToDevice);
	cudaFree(projVector);

	float* VolumeBackProj_device;
	cudaMalloc((void**)&VolumeBackProj_device, volumeWidth * volumeDepth * volumeHeight * sizeof(float));
	cudaMemcpy(VolumeBackProj_device, VolumeBackProj, volumeWidth * volumeDepth * volumeHeight * sizeof(float), cudaMemcpyHostToDevice);
	
	
	//cudaFree(VolumeBackProj);
	
	clock_t start, end;
	double time_used;

	
	// 这里放入你要计时的代码

	for (int iteration = 0; iteration < IterNum; iteration++) {
		cudaTextureDesc     texDescr_vol;
		memset(&texDescr_vol, 0, sizeof(cudaTextureDesc));
		texDescr_vol.normalizedCoords = false;
		texDescr_vol.filterMode = cudaFilterModeLinear;
		texDescr_vol.addressMode[0] = cudaAddressModeBorder;
		texDescr_vol.addressMode[1] = cudaAddressModeBorder;
		texDescr_vol.addressMode[2] = cudaAddressModeBorder;
		texDescr_vol.readMode = cudaReadModeElementType;
		const cudaChannelFormatDesc channelDesc_vol = cudaCreateChannelDesc<float>();
		cudaArray* d_cube_array_vol;
		cudaMalloc3DArray(&d_cube_array_vol, &channelDesc_vol, make_cudaExtent(volumeWidth, volumeDepth, volumeHeight));
		cudaMemcpy3DParms copyParams_vol = { 0 };
		copyParams_vol.dstArray = d_cube_array_vol;
		copyParams_vol.extent = make_cudaExtent(volumeWidth, volumeDepth, volumeHeight);
		copyParams_vol.kind = cudaMemcpyDeviceToDevice;
		cudaResourceDesc    texRes_vol;
		memset(&texRes_vol, 0, sizeof(cudaResourceDesc));
		texRes_vol.resType = cudaResourceTypeArray;

		// 将projection数据绑定到纹理内存
		cudaTextureDesc     texDescr_prj;
		memset(&texDescr_prj, 0, sizeof(cudaTextureDesc));
		texDescr_prj.normalizedCoords = false;
		texDescr_prj.filterMode = cudaFilterModeLinear;
		texDescr_prj.addressMode[0] = cudaAddressModeBorder;
		texDescr_prj.addressMode[1] = cudaAddressModeBorder;
		texDescr_prj.addressMode[2] = cudaAddressModeBorder; texDescr_prj.readMode = cudaReadModeElementType;
		const cudaChannelFormatDesc channelDesc_prj = cudaCreateChannelDesc<float>();
		cudaArray* d_cube_array_proj;
		cudaMalloc3DArray(&d_cube_array_proj, &channelDesc_prj, make_cudaExtent(projWidth, projHeight, anglenums));
		cudaMemcpy3DParms copyParams_prj = { 0 };
		copyParams_prj.dstArray = d_cube_array_proj;
		copyParams_prj.extent = make_cudaExtent(projWidth, projHeight, anglenums);
		copyParams_prj.kind = cudaMemcpyDeviceToDevice;
		cudaResourceDesc    texRes_prj;
		memset(&texRes_prj, 0, sizeof(cudaResourceDesc));
		texRes_prj.resType = cudaResourceTypeArray;
		cout << "iteration:" << iteration << endl;
		//auto start_time = std::chrono::system_clock::now();
		start = clock();
		// 将volume数据绑定到纹理内存
		cudaTextureObject_t tex_vol;
		copyParams_vol.srcPtr = make_cudaPitchedPtr(volume_device, volumeWidth * sizeof(float), volumeWidth, volumeDepth);
		cudaMemcpy3D(&copyParams_vol);
		texRes_vol.res.array.array = d_cube_array_vol;
		cudaCreateTextureObject(&tex_vol, &texRes_vol, &texDescr_vol, NULL);

		// 初始化投影用内存块,纹理内存对象使用几何体
		const dim3 blockSize_proj = dim3(BLOCK_X, BLOCK_Y);
		const dim3 gridSize_proj = dim3((projWidth + blockSize_proj.x - 1) / blockSize_proj.x, (projHeight + blockSize_proj.y - 1) / blockSize_proj.y);
		//正投影函数,调用数据为在设备端的数据
		
		for (int angle = 0; angle < anglenums; angle = angle +1) {
			
			fproj << < gridSize_proj, blockSize_proj >> > (sinogram_real_device, residual_device, projVector_device, volumeSize_device, projSize_device,
				volbiasz, dSampleInterval, dSliceInterval, anglenums, tex_vol, angle);
		}
		cudaDestroyTextureObject(tex_vol);
		// 执行反投影操作

		cudaTextureObject_t tex_proj;
		copyParams_prj.srcPtr = make_cudaPitchedPtr(residual_device, projWidth * sizeof(float), projWidth, projHeight);
		cudaMemcpy3D(&copyParams_prj);
		texRes_prj.res.array.array = d_cube_array_proj;
		cudaCreateTextureObject(&tex_proj, &texRes_prj, &texDescr_prj, NULL);
		//cudaFreeArray(d_cube_array_proj);
		// 初始化投影用内存块
		const dim3  blockSize_vol = dim3(BLOCK_X, BLOCK_Y);
		const dim3 gridSize_vol = dim3((volumeWidth + blockSize_vol.x - 1) / blockSize_vol.x, (volumeDepth + blockSize_vol.y - 1) / blockSize_vol.y);

		for (int angle = 0; angle < anglenums; angle=angle+1) {
			bproj << < blockSize_vol, gridSize_vol >> > (VolumeBackProj_device, residual_device, volumeSize_device, 
				projSize_device, projVector_device, volbiasz, dSampleInterval, dSliceInterval, anglenums, tex_proj, angle);
		}

		// 线性变换更新volume
		linetrans << < blockSize_vol, gridSize_vol >> > (volume_device, VolumeBackProj_device, volumeSize_device, miu);
		// relu变换更新volume

		//relutrans << <blockSize_vol, gridSize_vol >> > (volume_device, volumeSize_device);
		cudaFreeArray(d_cube_array_proj);
		cudaFreeArray(d_cube_array_vol);
		cudaDestroyTextureObject(tex_proj);

		if (iteration % 50 == 49) miu *= beta;

		// 计算时间间隔
		// 输出时间间隔（分钟）
		if (iteration%50== 49) {
			cudaMemcpy(VolumeBackProj, volume_device, volumeWidth * volumeDepth * volumeHeight * sizeof(float), cudaMemcpyDeviceToHost);
			//string filename1 = "E:\\WJC_train\\all_label\\label1mmB30L067.raw";
			string times = to_string(iteration);
			string filename1 = "SART360shepplogan"+times+"sparse" + ".raw";
			fp = fopen(filename1.c_str(), "wb");
			if (fp != nullptr)
				fwrite(VolumeBackProj, volumeWidth * volumeDepth * sizeof(float), volumeHeight, fp);
			else {
				cerr << "无法打开文件" << std::endl;
			}
			fclose(fp);
		}
		end = clock();

		time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

		printf("Elapsed time: %f seconds\n", time_used);
	}
	
	cudaFree(VolumeBackProj);
	cudaFree(sinogram_proj_device);
	cudaFree(sinogram_real_device);
	cudaFree(volume_device);
	cudaFree(residual_device);
	
	cudaFree(volumeSize_device);
	cudaFree(projSize_device);
	cudaFree(projVector_device);
	
	return 0;
}

