#include <stdio.h>
#include "CL/cl.h"

#define DATA_SIZE 10

/*
・OpenCL
CPUやGPUで非同期かつ並列で処理するためのフレームワーク。
マルチコア環境で能力を最大限に引き出すことを目的としている。

・プラットフォーム
GPUのメーカーが用意したOpenCLへの対応環境。
GPUごとにOpenCLへの対応は異なっているため、それぞれのGPUで異なる環境をもっている。複数のCPU, GPUがある場合、そのCPU, GPUごとにプラットフォームが異なる場合がある。

・デバイス
GPUかCPUのこと。
ひとつのPCに複数のCPU, GPUがある場合（マルチコア環境の場合）、そのすべてを活用するために一覧として取得する必要がある。そのときの単位として扱われる。

・コンテキスト
ひとつ以上のデバイスをまとめるオブジェクト。OpenCLに必要なGPUを動かす命令、メモリ領域などは全てこれを通して作る。
裏でデバイスを制御する黒幕、といったところ。

・コマンド
GPUに対する操作をカプセル化したオブジェクト。
GPUに対する操作とは、データを書き込み、読み取り、データに対する計算などのことをさす。

・コマンドキュー
コマンドを制御するキュー。このキューを用いて、デバイスへとコマンドを逐次投入する。

・バッファ
GPU内に生成するメモリ領域のこと。このバッファに対して書き込み、読み取りが行われる。

・プログラム
デバイスで動作する処理のこと。プログラムを動かすときには、OpenCL Cという言語を使用する。

・カーネル
GPUで動かす関数のこと。
カーネルは関数であり、このカーネルを複数まとめたものがプログラム。
*/

const char *KernelSource =
    "__kernel void hello(__global float *input, __global float *output)\n"
    "{\n"
    " size_t id = get_global_id(0);\n"
    " output[id] = input[id] * input[id];\n"
    "}\n"
    "\n";

int main(void)
{
    cl_context context;
    cl_context_properties properties[3];
    cl_kernel kernel;
    cl_command_queue command_queue;
    cl_program program;
    cl_int err;
    cl_uint num_of_platforms = 0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_of_devices = 0;
    cl_mem input, output;
    size_t global;

    float inputData[DATA_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float results[DATA_SIZE] = {0};

    int i;

    // retreives a list of platforms available
    if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS)
    {
        printf("Unable to get platform_id\n");
        return 1;
    }

    // try to get a supported GPU device
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                       &num_of_devices) != CL_SUCCESS)
    {
        printf("Unable to get device_id\n");
        return 1;
    }

    // context properties list - must be terminated with 0
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties)platform_id;
    properties[2] = 0;

    // create a context with the GPU device
    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

    // create command queue using the context and device
    command_queue = clCreateCommandQueue(context, device_id, 0, &err);

    // create a program from the kernel source code
    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);

    // compile the program
    if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error building program\n");
        return 1;
    }

    // specify which kernel from the program to execute
    kernel = clCreateKernel(program, "hello", &err);

    // create buffers for the input and ouput
    input = clCreateBuffer(context, CL_MEM_READ_ONLY,
                           sizeof(float) * DATA_SIZE, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                            sizeof(float) * DATA_SIZE, NULL, NULL);

    // load data into the input buffer
    clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0,
                         sizeof(float) * DATA_SIZE, inputData, 0, NULL, NULL);

    // set the argument list for the kernel command
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    global = DATA_SIZE;

    // enqueue the kernel command for execution
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global,
                           NULL, 0, NULL, NULL);
    clFinish(command_queue);

    // copy the results from out of the output buffer
    clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0,
                        sizeof(float) * DATA_SIZE, results, 0, NULL, NULL);

    // print the results
    printf("output: ");
    for (i = 0; i < DATA_SIZE; i++)
    {
        printf("%f ", results[i]);
    }
    printf("\n");

    // cleanup - release OpenCL resources
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}