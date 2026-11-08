#pragma once

// CUDA runtime shim for the paging runtime (moE pager, doorbell, predictor).
// The tensor backends themselves come from ggml; only this layer speaks cuda*.
// ggml's vendors/hip.h is deliberately not used here: it pulls in hipBLAS and
// defines __CUDA_ARCH__ 1300, which would enable the NVIDIA PTX paths below.

#if defined(GGML_USE_HIP)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef hipError_t  cudaError_t;
typedef hipStream_t cudaStream_t;
typedef hipEvent_t  cudaEvent_t;

#define cudaSuccess                           hipSuccess
#define cudaErrorInvalidValue                 hipErrorInvalidValue
#define cudaErrorMemoryAllocation             hipErrorOutOfMemory
#define cudaGetErrorString                    hipGetErrorString
#define cudaGetLastError                      hipGetLastError
#define cudaGetDevice                         hipGetDevice
#define cudaSetDevice                         hipSetDevice
#define cudaDeviceSynchronize                 hipDeviceSynchronize
#define cudaDeviceProp                        hipDeviceProp_t
#define cudaGetDeviceProperties               hipGetDeviceProperties

#define cudaMalloc                            hipMalloc
#define cudaFree                              hipFree
#define cudaMemset                            hipMemset
#define cudaMemsetAsync                       hipMemsetAsync

#define cudaMemcpy                            hipMemcpy
#define cudaMemcpyAsync                       hipMemcpyAsync
#define cudaMemcpyHostToDevice                hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost                hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice              hipMemcpyDeviceToDevice
#define cudaMemcpyDefault                     hipMemcpyDefault

#define cudaHostAlloc                         hipHostMalloc
#define cudaHostAllocDefault                  hipHostMallocDefault
#define cudaHostAllocPortable                 hipHostMallocPortable
#define cudaHostAllocMapped                   hipHostMallocMapped
#define cudaHostAllocWriteCombined            hipHostMallocWriteCombined
#define cudaFreeHost                          hipHostFree
#define cudaHostGetDevicePointer              hipHostGetDevicePointer
#define cudaHostRegister                      hipHostRegister
#define cudaHostRegisterDefault               hipHostRegisterDefault
#define cudaHostRegisterPortable              hipHostRegisterPortable
#define cudaHostUnregister                    hipHostUnregister

#define cudaStreamCreate                      hipStreamCreate
#define cudaStreamCreateWithFlags             hipStreamCreateWithFlags
#define cudaStreamDestroy                     hipStreamDestroy
#define cudaStreamSynchronize                 hipStreamSynchronize
#define cudaStreamWaitEvent                   hipStreamWaitEvent
#define cudaStreamNonBlocking                 hipStreamNonBlocking
#define cudaStreamIsCapturing                 hipStreamIsCapturing
#define cudaStreamCaptureStatus               hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone           hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive         hipStreamCaptureStatusActive

#define cudaEventCreate                       hipEventCreate
#define cudaEventCreateWithFlags              hipEventCreateWithFlags
#define cudaEventRecord                       hipEventRecord
#define cudaEventSynchronize                  hipEventSynchronize
#define cudaEventElapsedTime                  hipEventElapsedTime
#define cudaEventDestroy                      hipEventDestroy
#define cudaEventDisableTiming                hipEventDisableTiming

#ifndef __shfl_down_sync
// HIP accepts 3 or 4 argument calls; wave32 uses the default width
#define __shfl_down_sync(mask, var, lane, ...) __shfl_down(var, lane)
#endif

#else

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#endif
