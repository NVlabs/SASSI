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
 * This example tracks memory divergence, and is based on case study II in,
 *
 *   "Flexible Software Profiling of GPU Architectures"
 *   Stephenson et al., ISCA 2015.
 *
 * The application code the user instruments should be instrumented with the
 * following SASSI flag: -Xptxas --sassi-inst-before="memory"
 *                       -Xptxas --sassi-before-args="mem-info"
 *  
\***********************************************************************************/

#include <assert.h>
#include <cupti.h>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include "sassi_intrinsics.h"
#include "sassi_lazyallocator.hpp"
#include <sassi/sassi-core.hpp>
#include <sassi/sassi-memory.hpp>

/// The number of bits we need to shift off to get the cache line address.
#define LINE_BITS   5

// The width of a warp.
#define WARP_SIZE   32

/// The counters that we will use to record our statistics.
__managed__ unsigned long long sassi_counters[WARP_SIZE + 1][WARP_SIZE + 1];


///////////////////////////////////////////////////////////////////////////////////
///
///  Lazily initialize the counters before the first kernel launch. 
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator counterInitializer(
  /* Initialize the counters. */
  []() {
    bzero(sassi_counters, sizeof(sassi_counters));
  }, 

  /* Dump the stats. */
  [](sassi::lazy_allocator::device_reset_reason reason) {
    FILE *rf = fopen("sassi-memdiverge.txt", "a");
    fprintf(rf, "Active x Diverged:\n");
    for (unsigned m = 0; m <= WARP_SIZE; m++) {
      fprintf(rf, "%d ", m);
      for (unsigned u = 0; u <= WARP_SIZE; u++)
	{
	  fprintf(rf, "%llu ", sassi_counters[m][u]);
	}
      fprintf(rf, "\n");
    }
    fprintf(rf, "\n");
    fclose(rf);
  }
);


///////////////////////////////////////////////////////////////////////////////////
/// 
/// This is the function that will be inserted before every memory operation. 
/// 
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_before_handler(SASSIBeforeParams *bp, SASSIMemoryParams *mp)
{
  if (bp->GetInstrWillExecute())
  {
    intptr_t addrAsInt = mp->GetAddress();
    // Don't look at shared or local memory.
    if (__isGlobal((void*)addrAsInt)) { 
      // The number of unique addresses across the warp 
      unsigned unique = 0;   // for the instrumented instruction.

      // Shift off the offset bits into the cache line.
      intptr_t lineAddr =  addrAsInt >> LINE_BITS;

      int workset = __ballot(1);
      int firstActive = __ffs(workset) - 1;
      int numActive = __popc(workset);
      while (workset) {
	// Elect a leader, get its line, see who all matches it.
	int leader = __ffs(workset) - 1;
	intptr_t leadersAddr = __broadcast<intptr_t>(lineAddr, leader);
	int notMatchesLeader = __ballot(leadersAddr != lineAddr);

	// We have accounted for all values that match the leader's.
	// Let's remove them all from the workset.
	workset = workset & notMatchesLeader;
	unique++;
	assert(unique <= 32);
      }

      assert(unique > 0 && unique <= 32);

      // Each thread independently computed 'numActive' and 'unique'.
      // Let's let the first active thread actually tally the result.
      int threadsLaneId = get_laneid();
      if (threadsLaneId == firstActive) {
	atomicAdd(&(sassi_counters[numActive][unique]), 1LL);
      }
    }
  }
}
