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

 Tests out the dictionary.
\****************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <set>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <nvfunctional>
#include <sys/time.h>
#include "../sassi_dictionary.hpp"
#include "../sassi_intrinsics.h"
#include "../sassi_lazyallocator.hpp"

typedef sassi::dictionary<int, unsigned> IUMap;

__managed__ IUMap *map;

std::map<std::string, int> kCountMap;  // count number of kernel invocations
bool verbose = false;

////////////////////////////////////////////////////////////////////////////////////
//
//  Let's test out the lazy allocator while we are at it, which will let us
//  allocate our data structures before the first kernel launch.
//
////////////////////////////////////////////////////////////////////////////////////
static void mapAllocate() {
  map = new IUMap(5800079, 128);
}

////////////////////////////////////////////////////////////////////////////////////
//
// User defined host funtion to be executed on kernel entry.
// User has access to const CUpti_CallbackData* cbInfo. 
//
////////////////////////////////////////////////////////////////////////////////////
static void onKernelEntry(const CUpti_CallbackData* cbInfo) {
	std::string kName = cbInfo->symbolName;
	if (kCountMap.find(kName) != kCountMap.end()) { // kernel name found in kCountMap
		kCountMap[kName] += 1;
	} else { // kernel if seen for the first time
		kCountMap[kName] = 1;
	}
		
	if (verbose) {
  	printf("\nKernelEntry: Name=%s, Invocation count=%d\n", kName.c_str(), kCountMap[kName]);
	}
}

////////////////////////////////////////////////////////////////////////////////////
//
// User defined host funtion to be executed on every kernel exit.
// User has access to const CUpti_CallbackData* cbInfo. 
//
////////////////////////////////////////////////////////////////////////////////////
static void onKernelExit(const CUpti_CallbackData* cbInfo) {
	cudaError_t * error = (cudaError_t*) cbInfo->functionReturnValue;
	if ( (*error) != cudaSuccess ) {
		printf("Kernel Exit Error: %d", (*error));
	}

	if (verbose) {
  	printf("KernelExit: Name=%s, Return value=%d\n", cbInfo->symbolName, *error);
	}
}

////////////////////////////////////////////////////////////////////////////////////
//
// This user defined host function will be called after all kernels have been executed.
// This exmaple prints the list of kernel names that were executed during a particular 
// run along with the number of invocations.
//
////////////////////////////////////////////////////////////////////////////////////
static void finalize() {
	std::map<std::string, int>::iterator it;
	for(it=kCountMap.begin(); it!=kCountMap.end(); ++it) {
	 printf("Kernel Name: %s, Num invocations: %d\n", it->first.c_str(),  it->second);
	}
}

static sassi::lazy_allocator mapAllocator(mapAllocate, finalize, onKernelEntry, onKernelExit);



////////////////////////////////////////////////////////////////////////////////////
//
//  Inserts a key from 'random' and sets the value of the associated node to
//  the thread's tid.
//
////////////////////////////////////////////////////////////////////////////////////
double computeElapsed(timeval &t1, timeval &t2)
{
  return ((t2.tv_sec - t1.tv_sec) * 1000.0)
    + ((t2.tv_usec - t1.tv_usec) / 1000.0);
}

////////////////////////////////////////////////////////////////////////////////////
//
//  Inserts a key from 'random' and sets the value of the associated node to
//  the thread's tid.
//
////////////////////////////////////////////////////////////////////////////////////
__global__ void random_key_tid_value_kernel(int *random)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned *t = map->getOrInit(random[tid], [&tid](unsigned *v) {
      *v = tid;
    });
  assert (t != NULL);
}

////////////////////////////////////////////////////////////////////////////////////
//
//  Inserts a key from 'random' and increments the value of the associated node.
//
////////////////////////////////////////////////////////////////////////////////////
__global__ void agg_kernel(int *random)
{
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned *t = map->getOrInit(random[tid], [&tid](unsigned *v) {
      *v = 0;
    });
  assert (t != NULL);
  atomicAdd(t, 1);
}

////////////////////////////////////////////////////////////////////////////////////
//
//  Inserts M * N random nodes.
//
////////////////////////////////////////////////////////////////////////////////////
bool randomInsertTest(unsigned M, unsigned N, double &kt)
{
  std::vector<int> rngs(N * M);
  int *dev_rngs;
  timeval t1, t2;
  
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_rngs, M * N * sizeof(int)));

  srand48(0);
  for (unsigned i = 0; i < M * N; i++) rngs[i] = lrand48();

  CHECK_CUDA_ERROR(cudaMemcpy(dev_rngs, &rngs[0], M * N * sizeof(int), cudaMemcpyHostToDevice));

  gettimeofday(&t1, NULL);
  random_key_tid_value_kernel<<<M, N>>>(dev_rngs);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  kt = computeElapsed(t1, t2);
  CHECK_CUDA_ERROR(cudaFree(dev_rngs));

  bool failed = false;
  std::set<unsigned> values;
  map->map([&](int &k, unsigned &v) {
      failed |= (k != rngs[v]);
      values.insert(v);
    });

  if (map->size() != values.size()) {
    printf("size %zu != M, %d\n", values.size(), map->size());
    failed = true;
  }

  for (unsigned v : values) {
    failed |= (v > (M * N));
  }

  return !failed;
}

////////////////////////////////////////////////////////////////////////////////////
//
//  Inserts M * N nodes in sorted order.
//
////////////////////////////////////////////////////////////////////////////////////
bool linearInsertTest(unsigned M, unsigned N, double &kt)
{
  std::vector<int> rngs(N * M);
  int *dev_rngs;
  timeval t1, t2;
  
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_rngs, M * N * sizeof(int)));
  
  for (unsigned i = 0; i < M * N; i++) rngs[i] = i;

  CHECK_CUDA_ERROR(cudaMemcpy(dev_rngs, &rngs[0], M * N * sizeof(int), cudaMemcpyHostToDevice));

  gettimeofday(&t1, NULL);
  random_key_tid_value_kernel<<<M, N>>>(dev_rngs);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  kt = computeElapsed(t1, t2);
  CHECK_CUDA_ERROR(cudaFree(dev_rngs));

  bool failed = false;
  std::set<unsigned> values;
  map->map([&](int &k, unsigned &v) {
      failed |= ((k != rngs[v]) || ((unsigned)k != v));
      values.insert(v);
    });
  
  failed |= map->size() != values.size();
  failed |= map->size() != (M * N);
  for (unsigned i = 0; i < M * N; i++) {
    failed |= (values.find(i) == values.end());
  }

  return !failed;
}

////////////////////////////////////////////////////////////////////////////////////
//
//  Inserts at most modulus unique keys in linear order.  If M * N > modulus,
//  then we will insert some duplicate keys.
//
////////////////////////////////////////////////////////////////////////////////////
bool linearAggTest(int M, int N, unsigned modulus, double &kt)
{
  std::vector<int> rngs(N * M);
  std::vector<unsigned> sums(modulus);
  int *dev_rngs;
  timeval t1, t2;
  
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_rngs, M * N * sizeof(int)));
  
  for (int i = (M * N) - 1; i >= 0; i--) {
    int m = i % modulus;
    rngs[i] = m;
    sums[m]++;
  }

  CHECK_CUDA_ERROR(cudaMemcpy(dev_rngs, &rngs[0], M * N * sizeof(int), cudaMemcpyHostToDevice));

  gettimeofday(&t1, NULL);
  agg_kernel<<<M, N>>>(dev_rngs);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  kt = computeElapsed(t1, t2);
  CHECK_CUDA_ERROR(cudaFree(dev_rngs));

  bool failed = false;
  std::set<int> values;
  map->map([&](int &k, unsigned &v) {
      failed |= (sums[k] != v);
      values.insert(k);
    });

  failed |= (modulus != values.size());
  failed |= (modulus != map->size());
  for (unsigned i = 0; i < modulus; i++) {
    failed |= (values.find(i) == values.end());
  }
  
  return !failed;
}

////////////////////////////////////////////////////////////////////////////////////
//
//  Clears the hash table so we can run the next test.
//
////////////////////////////////////////////////////////////////////////////////////
void clearTable()
{
  cudaDeviceSynchronize();
  map->clear();
}

////////////////////////////////////////////////////////////////////////////////////
//
//  Run the tests.
//
////////////////////////////////////////////////////////////////////////////////////
int main(void) 
{
  int const tests[][2] = {{1, 4}, {1, 20}, {1, 256}, {1, 1024},
			  {4, 1}, {19, 1}, {256, 1}, {1024, 1},
			  {4, 4}, {256, 256}, {300, 300},
			  {512, 512},{1024, 1024},{1900, 1024},
			  {2048, 1}, {2048, 1024},{1000000,2},
			  {5679, 234}, {9876, 186}};

  int passed = 0;
  int failed = 0;
  int total = 0;

  for (unsigned i = 0; i < sizeof(tests)/sizeof(int[2]); i++)
    {
      double kernel;

      int M = tests[i][0];
      int N = tests[i][1];
      printf("Testing %dx%d kernels\n", M, N);

      printf("1. Inserting %d random keys: ", M * N); fflush(stdout);
      bool rit = randomInsertTest(M, N, kernel);
      printf("%s (%f kernel)\n", 
	     rit ? " PASSED" : "FAILED", kernel);
      fflush(stdout);
      clearTable();

      printf("2. Inserting %d linear keys: ", M * N); fflush(stdout);
      bool lit = linearInsertTest(M, N, kernel);
      printf("%s (%f kernel)\n",
	     lit ? " PASSED" : "FAILED", kernel);
      fflush(stdout);
      clearTable();

      printf("3. Inserting %d items with repetition: ", M * N); fflush(stdout);
      bool agg = linearAggTest(M, N, min(M, N), kernel);
      printf("%s (%f kernel)\n",
	     agg ? " PASSED" : "FAILED", kernel);
      fflush(stdout);
      clearTable();

      total += 3;
      if (rit) passed++; else failed++;
      if (lit) passed++; else failed++;
      if (agg) passed++; else failed++;
    }

  printf("Total tests : %d\n", total);
  printf("Total passed: %d\n", passed);
  printf("Total failed: %d\n", failed);
}
