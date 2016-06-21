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
 * In addition, be sure to link your application with flags necessary to 
 * hijack "main" and "exit".  You can trivially do this using GNU tools with
 *
 *       -Xlinker "--wrap=main" -Xlinker "--wrap=exit"
 *
 * This will cause calls to main and exit to be replaced by calls to 
 * __wrap_exit(int status) and __wrap_main(int argc, char **argv), which we have
 * defined below.  This allows us to do initialization and finalization without
 * having to worry about object constructor and destructor orders.
 *
 * This version of the library also lets us correlate SASS location to the
 * corresponding CUDA source locations.  To use this feature, you must 
 * compile your application with the "-lineinfo" option.
 *
 * See the branch example in example/Makfile for all the flags you should use.
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
#include "sassi_srcmap.hpp"
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

static sassi::src_mapper *sassiMapper;

// The actual dictionary of counters, where the key is a branch's PC, and
// the value is the set of counters associated with it.
static __managed__ sassi::dictionary<uint64_t, BranchCounter> *sassiStats;

// Convert the SASSIBranchType to a string that we can print.  See the
// CUDA binary utilities webpage for more information about these types.
const char *SASSIBranchTypeAsString[] = {
  "BRX", "BRA", "RET", "EXIT", "SYNC", "OTHER"
};


///////////////////////////////////////////////////////////////////////////////////
///
///  Collect the stats and print them out before the device counters are reset.
///
///////////////////////////////////////////////////////////////////////////////////
static void sassi_finalize(__attribute__((unused)) sassi::cupti_wrapper *wrapper, 
			   __attribute__((unused)) const CUpti_CallbackData *cb)
{
  // This function will be called either when 1) the device is reset, or 2) the
  // the program is about to exit.  Let's check to see whether the sassiStats
  // map is still valid.  For instance, the user could have reset the device 
  // before the program exited, which would essentially invalidate all device
  // data. (In fact, explicitly reseting the device before program exit is
  // considered best practice.)
  if (sassiMapper->is_device_state_valid())
  {
    FILE *fRes = fopen("sassi-branch.txt", "w");
    
    fprintf(fRes, "%-16.16s %-10.10s %-10.10s %-10.10s %-10.10s %-10.10s %-8.8s %-8.8s Location\n",
	    "Address", "Total/32", "Dvrge/32", "Active", "Taken", "NTaken", 
	    "Type", ".U");

    // Get the SASS PUPC to source code line mapping.
    auto const locMapper = sassiMapper->get_location_map();
    
    sassiStats->map([fRes,&locMapper](uint64_t& pupc, BranchCounter& val) {
	assert(val.address == pupc);
	
	fprintf(fRes, "%-16.16" PRIx64 
		" %-10.llu %-10.llu %-10.llu %-10.llu %-10.llu %-8.4s %-8.d ",
		pupc,
		val.totalBranches, 
		val.divergentBranches,
		val.activeThreads,
		val.takenThreads,
		val.takenNotThreads,
		SASSIBranchTypeAsString[val.branchType],
		val.taggedUnanimous
		);      

	// See if there is a source code mapping for this PUPC.  If you 
	// compiled your code with "-lineinfo" there should be a valid
	// mapping.
	auto it = locMapper.find(pupc);
	if (it != locMapper.end()) {
	  fprintf(fRes, "%s, line %d\n", it->second.file_name->c_str(), it->second.line_num);
	} else {
	  fprintf(fRes, "\n");
	}
      });
  
    fclose(fRes);
  }
}

///////////////////////////////////////////////////////////////////////////////////
/// 
///  We will compile our application using ld's --wrap option, which in this
///  case lets us replace calls to "exit" with calls to "__wrap_exit".  See
///  the make target "ophist-fermi" in ./example/Makefile to see how this
///  is done.
///
///  This should allow us to perform CUDA operations before the CUDA runtime
///  starts shutting down.  In particular, we want to copy our
///  "dynamic_instr_counts" off the device.  If we used UVM, this would happen
///  automatically for us.  But since we don't have the luxury of using UVM
///  for Fermi, we have to make sure that the CUDA runtime is still up and
///  running before trying to issue a cudaMemcpy.  Hence these shenanigans.
/// 
///////////////////////////////////////////////////////////////////////////////////
extern "C" void __real_exit(int status);
extern "C" void __wrap_exit(int status)
{
  sassi_finalize(NULL, NULL);
  __real_exit(status);
}

///////////////////////////////////////////////////////////////////////////////////
/// 
///  For programs that don't call exit explicitly, let's catch the fallthrough.
/// 
///////////////////////////////////////////////////////////////////////////////////
extern "C" int __real_main(int argc, char **argv);
extern "C" int __wrap_main(int argc, char **argv)
{
  // Initialize a src_mapper to give us SASS PC->CUDA line mappings.
  sassiMapper = new sassi::src_mapper();

  // Initialize a hashmap to keep track of statistics of branches.  The key
  // is the PC, the value is a BranchCounter.
  sassiStats = new sassi::dictionary<uint64_t, BranchCounter>();

  // Whenever the device is reset, be sure to print out the counters before
  // they are clobbered.
  sassiMapper->register_callback(sassi::cupti_wrapper::event_type::DEVICE_RESET, 
				 sassi::cupti_wrapper::callback_before,
				 sassi_finalize);

  int ret = __real_main(argc, argv);
  sassi_finalize(NULL, NULL);
  return ret;
}

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
    uint64_t instAddr = bp->GetPUPC();
    
    // Looks up the counters associated with 'instAddr', but if no such entry
    // exits, initialize the counters in the lambda.
    BranchCounter *stats = (*sassiStats).getOrInit(instAddr, [instAddr,brp](BranchCounter* v) {
	v->address = instAddr;
	v->branchType = brp->GetType();
	v->taggedUnanimous = brp->IsUnanimous();
      });

    // Why not sanity check the hash map?
    assert(stats->address == instAddr);
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



