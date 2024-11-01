#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Backpro.h"
#include "Weight.h"
#include "conv.h"
#include <vector>
#include <stdio.h>
#include <cmath>
#include<iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <cstring>

using namespace std;
#define pi  3.1415926
//编写一维卷积函数

int main()
{
    // 设置射线扫描物体参数
    const int volumeWidth = 600;
    const int volumeHeight = 100;
    const int volumeDepth = 600;
    float volbiasz =-900.0f;
    float dSampleInterval = 1.0f;
    float dSliceInterval = 1.0f;
    // 用一个长度为3的int类型数组存储几何体几何参数
    int* volumeSize = new int[3];
    volumeSize[0] = volumeWidth;
    volumeSize[1] = volumeDepth;
    volumeSize[2] = volumeHeight;
    // 设置探测器板参数
    const int projWidth = 1600;
    const int projHeight = 1600;
    // 用一个长度为2的int类型数组存储探测器几何参数
    int* projSize = new int[2];
    projSize[0] = projWidth;
    projSize[1] = projHeight;
    // 设置投影角度总数
    const int anglenums = 32;
    // 设置残差更新初始比率
    // 设置GPU内存块值
    const int BLOCK_X_proj = 32;
    const int BLOCK_Y_vol = 16;
    // 初始化投影用内存块
    const dim3 blockSize_proj = dim3(BLOCK_X_proj, BLOCK_X_proj);
    const dim3 gridSize_proj = dim3((projWidth + blockSize_proj.x - 1) / blockSize_proj.x, (projHeight + blockSize_proj.y - 1) / blockSize_proj.y);
    const dim3  blockSize_vol = dim3(BLOCK_Y_vol, BLOCK_Y_vol);
    const dim3 gridSize_vol = dim3((volumeWidth + blockSize_vol.x - 1) / blockSize_vol.x, (volumeDepth + blockSize_vol.y - 1) / blockSize_vol.y);


    //读取matlab上存储的geometry文件
    float* projVector = new float[12 * anglenums];
    memset(projVector, 0, 12 * anglenums * sizeof(float));
    const char* filename_geo = "proj_vec_ss.raw";
    //const char* filename_geo = "geometry.raw";
    //const char* filename_geo = "geometry_sparse.raw";
    FILE* fp2 = nullptr;
    if ((fp2 = fopen(filename_geo, "rb")) == nullptr) {
        printf("无法打开 %s 文件 ", filename_geo);
        return false;
    }

    fread(projVector, sizeof(float), 12 * anglenums, fp2);
    fclose(fp2);
    //读取使用matlab生成的3d shepp_logan 投影数据
    float* sinogram = new float[projWidth * projHeight * anglenums];
    memset(sinogram, 0, projWidth * projHeight * anglenums * sizeof(float));
    const char* filename = "004_1.raw";
    //const char* filename = "sinogram_file.raw";
    //const char* filename = "sinogram_file_sparse.raw";
    FILE* fp = nullptr;
    if ((fp = fopen(filename, "rb")) == nullptr) {
        printf("无法打开 %s 文件 ", filename);
        return false;
    }
    fread(sinogram, sizeof(float), projWidth * projHeight * anglenums, fp);
    fclose(fp);
    // 初始化卷积后序列
    float* sinogram_pro = new float[projHeight * projWidth * anglenums];
    memset(sinogram_pro, 0, projHeight * projWidth * anglenums * sizeof(float));
    //初始化滤波器
    float* filter = (float*)malloc(projWidth * sizeof(float));
    memset(filter, 0, projWidth * sizeof(float));
    int mid_pos = round((projWidth - 1) / 2);
    // 生成设备端数组
    float* sinogram_device;
    cudaMalloc((void**)&sinogram_device, projWidth * projHeight * anglenums * sizeof(float));
    cudaMemcpy(sinogram_device, sinogram, projWidth * projHeight * anglenums * sizeof(float), cudaMemcpyHostToDevice);

    float* projVector_device;
    cudaMalloc((void**)&projVector_device, 12 * anglenums * sizeof(float));
    cudaMemcpy(projVector_device, projVector, 12 * anglenums * sizeof(float), cudaMemcpyHostToDevice);

    int* volumeSize_device;
    cudaMalloc((void**)&volumeSize_device, 3 * sizeof(int));
    cudaMemcpy(volumeSize_device, volumeSize, 3 * sizeof(int), cudaMemcpyHostToDevice);

    int* projSize_device;
    cudaMalloc((void**)&projSize_device, 2 * sizeof(int));
    cudaMemcpy(projSize_device, projSize, 2 * sizeof(int), cudaMemcpyHostToDevice);
    //cudaFree(projSize);

    float* sinogram_pro_device;
    cudaMalloc((void**)&sinogram_pro_device, projWidth * projHeight * anglenums * sizeof(float));
    cudaMemcpy(sinogram_pro_device, sinogram_pro, projWidth * projHeight * anglenums * sizeof(float), cudaMemcpyHostToDevice);

    string filename1 = "result";
    // 加权
    for (int angle = 0; angle < anglenums; angle = angle + 1) {
        WeightKernal << < blockSize_proj, gridSize_proj >> > (sinogram_device, projSize_device,volumeSize_device,projVector_device,angle,anglenums,dSampleInterval,dSliceInterval,volbiasz);
    }
    // begin convolution
    printf("Begin convolution");
    for (int i = 0; i < projWidth; i++) {
        int i_p = i - mid_pos;
        if (i_p == 0) {
            filter[i] = 1 / 4.0;
        }
        else if (i_p % 2) {
            filter[i] = -1 / (i_p * i_p * pi * pi);
        }
        else
            filter[i] = 0;
    }

    float* filter_device;
    cudaMalloc((void**)&filter_device, projWidth * sizeof(float));
    cudaMemcpy(filter_device, filter, projWidth * sizeof(float), cudaMemcpyHostToDevice);
    //cudaFree(filter);

    const int BLOCK_CONV_X = 4;
    const int BLOCK_CONV_Y = 32;
    // 初始化投影用内存块
    const dim3 blockSize_conv_proj = dim3(BLOCK_CONV_X, BLOCK_CONV_Y);
    const dim3 gridSize_conv_proj = dim3((anglenums + blockSize_conv_proj.x - 1) / blockSize_conv_proj.x, (projWidth + blockSize_conv_proj.y - 1) / blockSize_conv_proj.y);

    // 新建卷积序列中间量
    // 卷积滤波
    Conv_Kernal << <blockSize_conv_proj, gridSize_conv_proj >> > (sinogram_device, projWidth, filter_device, projWidth, sinogram_pro_device, anglenums, projWidth, projHeight);
    // 反投影

    // 初始化三维反投影张量，用一维数组来存储
    float* shepplogan3D_reconstruct = (float*)malloc(volumeWidth * volumeDepth * volumeHeight * sizeof(float));
    memset(shepplogan3D_reconstruct, 0, volumeWidth * volumeDepth * volumeHeight * sizeof(float));
    //初始化gpu数据
    float* shepplogan3D_reconstruct_device;
    //设备上分配内存以存储指针数组
    cudaMalloc((void**)&shepplogan3D_reconstruct_device, volumeWidth * volumeDepth * volumeHeight * sizeof(float));
    //将指针数组传输到设备
    cudaMemcpy(shepplogan3D_reconstruct_device, shepplogan3D_reconstruct, volumeWidth * volumeDepth * volumeHeight * sizeof(float), cudaMemcpyHostToDevice);
    // 将sinogram_pro转为纹理内存
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
    cudaTextureObject_t tex_proj;
    copyParams_prj.srcPtr = make_cudaPitchedPtr(sinogram_pro_device, projWidth * sizeof(float), projWidth, projHeight);
    cudaMemcpy3D(&copyParams_prj);
    texRes_prj.res.array.array = d_cube_array_proj;
    cudaCreateTextureObject(&tex_proj, &texRes_prj, &texDescr_prj, NULL);

    for (int angle = 0; angle < anglenums; angle++) {
        back_projection << <gridSize_vol, blockSize_vol >> > (shepplogan3D_reconstruct_device, projVector_device, volumeSize_device, projSize_device, volbiasz,
            dSampleInterval, dSliceInterval, tex_proj, angle, anglenums);
    }
    cudaMemcpy(shepplogan3D_reconstruct, shepplogan3D_reconstruct_device, volumeWidth * volumeDepth * volumeHeight * sizeof(float), cudaMemcpyDeviceToHost);


    filename1 += to_string(100) + ".raw";
    fp = fopen(filename1.c_str(), "wb");
    if (fp != nullptr)
        fwrite(shepplogan3D_reconstruct, volumeWidth * volumeDepth * sizeof(float), volumeHeight, fp);
    else {
        cerr << "无法打开文件" << std::endl;
    }
    fclose(fp);


    // 释放相关内存
    //释放纹理内存及变
    /*
    cudaFreeArray(d_cube_array_proj);
    cudaDestroyTextureObject(tex_proj);
    cudaFree(shepplogan3D_reconstruct);
    cudaFree(projVector_device);
    cudaFree(shepplogan3D_reconstruct_device);
    cudaFree(sinogram_pro_device);
    cudaFree(projSize_device);
    cudaFree(projSize_device);
    cudaFree(volumeSize_device);
    cudaFree(sinogram);
    cudaFree(projVector);
    cudaFree(sinogram_pro);
    cudaFree(volumeSize);
    cudaFree(projSize);
    cudaFree(sinogram_device);
*/
}

