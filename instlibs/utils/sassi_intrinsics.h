/****************************************************************************\
 Copyright (c) 2015, NVIDIA open source projects
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of SASSI nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 Some functions that are common enough that they should be intrinsics.
\****************************************************************************/

#ifndef __INTRINSICS_H____
#define __INTRINSICS_H____

#include <cuda.h>

/////////////////////////////////////////////////////////////////////////////
//
//  Check for a CUDA error.
//
/////////////////////////////////////////////////////////////////////////////
#define CHECK_CUDA_ERROR(err)                        \
  if (err != cudaSuccess) {                          \
     printf("Error: %s\n", cudaGetErrorString(err)); \
  }

/////////////////////////////////////////////////////////////////////////////
//
//  Check for a CUPTI error.
//
/////////////////////////////////////////////////////////////////////////////
#define CHECK_CUPTI_ERROR(err, cuptifunc)                            \
  if (err != CUPTI_SUCCESS)                                          \
  {                                                                  \
     const char *errstr;                                             \
     cuptiGetResultString(err, &errstr);                             \
     printf ("%s:%d:Error %s for CUPTI API function '%s'.\n",        \
     __FILE__, __LINE__, errstr, cuptifunc);                         \
     exit(-1);                                                       \
  }

/////////////////////////////////////////////////////////////////////////////
//
//  Get a thread's CTA ID.
//
/////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ int4 get_ctaid(void) {
  int4 ret;
  asm("mov.u32 %0, %ctaid.x;" : "=r"(ret.x));
  asm("mov.u32 %0, %ctaid.y;" : "=r"(ret.y));
  asm("mov.u32 %0, %ctaid.z;" : "=r"(ret.z));
  return ret;
}

/////////////////////////////////////////////////////////////////////////////
//
//  Get the number of CTA ids per grid.
//
/////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ int4 get_nctaid(void) {
  int4 ret;
  asm("mov.u32 %0, %nctaid.x;" : "=r"(ret.x));
  asm("mov.u32 %0, %nctaid.y;" : "=r"(ret.y));
  asm("mov.u32 %0, %nctaid.z;" : "=r"(ret.z));
  return ret;
}

/////////////////////////////////////////////////////////////////////////////
//
//  Get a thread's SM ID.
//
/////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ unsigned int get_smid(void) {
     unsigned int ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

/////////////////////////////////////////////////////////////////////////////
//
//  Get a thread's warp ID.
//
/////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ unsigned int get_warpid(void) {
     unsigned int ret;
     asm("mov.u32 %0, %warpid;" : "=r"(ret) );
     return ret;
}

/////////////////////////////////////////////////////////////////////////////
//
//  Get a thread's lane ID.
//
/////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ unsigned int get_laneid(void) {
  unsigned int laneid;
  asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid));
  return laneid;
}  

/////////////////////////////////////////////////////////////////////////////
//
//  Returns true if the pointer points to shared memory.  Similar to the
//  isGlobal() CUDA intrinsic.
//
/////////////////////////////////////////////////////////////////////////////
__device__ unsigned int __isShared(const void *ptr)
{
  unsigned int ret;
  asm volatile ("{ \n\t"
                "    .reg .pred p; \n\t"
                "    isspacep.shared p, %1; \n\t"
                "    selp.u32 %0, 1, 0, p;  \n\t"
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
                "} \n\t" : "=r"(ret) : "l"(ptr));
#else
                "} \n\t" : "=r"(ret) : "r"(ptr));
#endif

  return ret;
}

/////////////////////////////////////////////////////////////////////////////
//
//  Returns true if the pointer points to local memory.  Similar to the
//  isGlobal() CUDA intrinsic.
//
/////////////////////////////////////////////////////////////////////////////
__device__ unsigned int __isLocal(const void *ptr)
{
  unsigned int ret;
  asm volatile ("{ \n\t"
                "    .reg .pred p; \n\t"
                "    isspacep.local p, %1; \n\t"
                "    selp.u32 %0, 1, 0, p;  \n\t"
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
                "} \n\t" : "=r"(ret) : "l"(ptr));
#else
                "} \n\t" : "=r"(ret) : "r"(ptr));
#endif

  return ret;
}

/////////////////////////////////////////////////////////////////////////////
//
//  Returns true if the pointer points to constant memory.  Similar to the
//  isGlobal() CUDA intrinsic.
//
/////////////////////////////////////////////////////////////////////////////
__device__ unsigned int __isConst(const void *ptr)
{
  unsigned int ret;
  asm volatile ("{ \n\t"
                "    .reg .pred p; \n\t"
                "    isspacep.const p, %1; \n\t"
                "    selp.u32 %0, 1, 0, p;  \n\t"
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
                "} \n\t" : "=r"(ret) : "l"(ptr));
#else
                "} \n\t" : "=r"(ret) : "r"(ptr));
#endif

  return ret;
}

/////////////////////////////////////////////////////////////////////////////
//
//  A semi-generic warp broadcast function.
//
/////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ T __broadcast(T t, int fromWhom)
{
  union {
    int32_t shflVals[sizeof(T)];
    T t;
  } p;
  
  p.t = t;
  #pragma unroll
  for (int i = 0; i < sizeof(T); i++) {
    int32_t shfl = (int32_t)p.shflVals[i];
    p.shflVals[i] = __shfl(shfl, fromWhom);
  }
  return p.t;
}

#endif
