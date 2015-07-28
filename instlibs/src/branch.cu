/*********************************************************************************** \
 * Copyright (c) 2015, NVIDIA open source projects
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * - Neither the name of SASSI nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * This is a SASSI instrumentation library for gathering branch statistics.  It 
 * corresponds to Case Study I in,
 *
 *   "Flexible Software Profiling of GPU Architectures"
 *   Stephenson et al., ISCA 2015.
 *  
 * The application code the user instruments should be instrumented with the
 * following SASSI flag: -Xptxas --sassi-inst-before="cond-branches" \
 *                       -Xptxas --sassi-before-args="cond-branch-info".
 *
\***********************************************************************************/

#define __STDC_FORMAT_MACROS
#include <assert.h>
#include <cupti.h>
#include <inttypes.h>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include "sassi_intrinsics.h"
#include "sassi_dictionary.hpp"
#include "sassi_lazyallocator.hpp"
#include <sassi/sassi-core.hpp>
#include <sassi/sassi-branch.hpp>


struct BranchCounter {
  uint64_t address;
  int32_t branchType;                    // The branch type.
  int32_t taggedUnanimous;               // Branch had .U modifier, so compiler knows...
  unsigned long long totalBranches;
  unsigned long long takenThreads;
  unsigned long long takenNotThreads;
  unsigned long long divergentBranches;   // Not all branches go the same way.
  unsigned long long activeThreads;       // Number of active threads.
};                                        


// The actual dictionary of counters, where the key is a branch's PC, and
// the value is the set of counters associated with it.
static __managed__ sassi::dictionary<uint64_t, BranchCounter> *sassi_stats;

// Convert the SASSIBranchType to a string that we can print.  See the
// CUDA binary utilities webpage for more information about these types.
const char *SASSIBranchTypeAsString[] = {
  "BRX", "BRA", "RET", "EXIT", "SYNC", "OTHER"
};

///////////////////////////////////////////////////////////////////////////////////
///
///  Print out the statistics.
///
///////////////////////////////////////////////////////////////////////////////////
static void sassi_finalize()
{
  cudaDeviceSynchronize();

  FILE *fRes = fopen("sassi-branch.txt", "w");

  fprintf(fRes, "%-16.16s %-10.10s %-10.10s %-10.10s %-10.10s %-10.10s %-8.8s %-8.8s\n",
	  "Address", "Total/32", "Dvrge/32", "Active", "Taken", "NTaken", 
	  "Type", ".U");
  
  sassi_stats->map([fRes](uint64_t& key, BranchCounter& val) {
      assert(val.address == key);
      fprintf(fRes, "%-16.16" PRIx64 
	      " %-10.llu %-10.llu %-10.llu %-10.llu %-10.llu %-8.4s %-8.d\n",
	      key,
	      val.totalBranches, 
	      val.divergentBranches,
	      val.activeThreads,
	      val.takenThreads,
	      val.takenNotThreads,
	      SASSIBranchTypeAsString[val.branchType],
	      val.taggedUnanimous);      
    });
  
  fclose(fRes);
}


///////////////////////////////////////////////////////////////////////////////////
///
///  Lazily allocate a dictionary before the first kernel launch. The dictionary 
///  will be available on the host and the device.  The allocator takes two 
///  functions: the first is an initialization function that's called once before 
///  any kernels run.  The next is a finalize function that's called before the 
///  program exits, or right before the device is reset.
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator mapAllocator([]() {
    sassi_stats = new sassi::dictionary<uint64_t, BranchCounter>();
  }, sassi_finalize);


///////////////////////////////////////////////////////////////////////////////////
//
/// This function will be inserted before every conditional branch instruction.
//
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_before_handler(SASSIBeforeParams *bp, SASSICondBranchParams *brp) 
{
  // Find out thread index within the warp.
  int threadIdxInWarp = get_laneid();

  // Get masks and counts of 1) active threads in this warp,
  // 2) threads that take the branch, and
  // 3) threads that do not take the branch.
  int active = __ballot(1);
  bool dir = brp->GetDirection();
  int taken = __ballot(dir == true);
  int ntaken = __ballot(dir == false);
  int numActive = __popc(active);
  int numTaken = __popc(taken);
  int numNotTaken = __popc(ntaken);
  bool divergent = (numTaken != numActive && numNotTaken != numActive);

  // The first active thread in each warp gets to write results.
  if ((__ffs(active)-1) == threadIdxInWarp) {
    // Get the address, we'll use it for hashing.
    uint64_t inst_addr = bp->GetPUPC();
    
    // Looks up the counters associated with 'inst_addr', but if no such entry
    // exits, initialize the counters in the lambda.
    BranchCounter *stats = (*sassi_stats).getOrInit(inst_addr, [inst_addr,brp](BranchCounter* v) {
	v->address = inst_addr;
	v->branchType = brp->GetType();
	v->taggedUnanimous = brp->IsUnanimous();
      });

    // Why not sanity check the hash map?
    assert(stats->address == inst_addr);
    assert(numTaken + numNotTaken == numActive);

    // Increment the various counters that are associated
    // with this instruction appropriately.
    atomicAdd(&(stats->totalBranches), 1ULL);
    atomicAdd(&(stats->activeThreads), numActive);
    atomicAdd(&(stats->takenThreads), numTaken);
    atomicAdd(&(stats->takenNotThreads), numNotTaken);
    atomicAdd(&(stats->divergentBranches), divergent);
  }
}



