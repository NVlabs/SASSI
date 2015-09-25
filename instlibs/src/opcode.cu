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
 * This example shows how to use SASSI to extract SASS opcodes and register
 * information.  It will dump a file that shows all of the instructions that
 * were executed at least once, and for each instruction, it will dump all 
 * of the registers that were sourced and defined.
 *
 * The application code the user instruments should be instrumented with the
 * following SASSI flags: -Xptxas --sassi-inst-before="all" \
 *                        -Xptxas --sassi-before-args="reg-info"
 *  
\***********************************************************************************/

#define __STDC_FORMAT_MACROS
#include <algorithm>
#include <assert.h>
#include <cupti.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include "sassi_intrinsics.h"
#include "sassi_dictionary.hpp"
#include "sassi_lazyallocator.hpp"
#include <sassi/sassi-core.hpp>
#include <sassi/sassi-regs.hpp>
#include <sassi/sassi-opcodes.h>


// A struct to record the opcode and registers used by each instruction.
struct SASSIInfo {
  SASSIInstrOpcode opcode;
  int32_t numGPRDstPtrs;
  int32_t numGPRSrcPtrs;
  int32_t GPRDsts[SASSI_NUMGPRDSTS];
  int32_t GPRSrcs[SASSI_NUMGPRSRCS]; 
  bool    CCDst;
  bool    CCSrc;
  int32_t PRDst;
  int32_t PRSrc;
  uint64_t pupc;
  unsigned long long weight;
};


/// The actual dictionary, declared as a UVM managed type.
static __managed__ sassi::dictionary<uint64_t, SASSIInfo> *sassi_stats;


///////////////////////////////////////////////////////////////////////////////////
/// 
///  Print out predicate usage.
/// 
///////////////////////////////////////////////////////////////////////////////////
static void print_predicates(FILE *resultsFile, int32_t mask)
{
  int i = 0;
  while (mask != 0) {
    if (mask & 0x1) {
      fprintf(resultsFile, "(p%d) ", i);
    }
    mask = mask >> 1;
    i++;
  }
 }

///////////////////////////////////////////////////////////////////////////////////
/// 
///  Prints out CC register usage.
/// 
///////////////////////////////////////////////////////////////////////////////////
static void print_cc(FILE *resultsFile, bool cc)
{
  if (cc) {
    fprintf(resultsFile, "(CC) ");
  } else {
    fprintf(resultsFile, "     ");
  }
}

///////////////////////////////////////////////////////////////////////////////////
/// 
///  Write out the statistics we've gathered.
/// 
///////////////////////////////////////////////////////////////////////////////////
static void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason)
{
  struct KVTuple {
    uint64_t pupc;
    SASSIInfo *v;
  };

  FILE *resultsFile = fopen("sassi-opcode-map.txt", "a");

  // Let's first sort the instructions according to PC.
  std::vector<KVTuple> ops;
  sassi_stats->map([&ops](uint64_t& key, SASSIInfo &val) {
      ops.push_back({key, &val});
    });

  std::sort(ops.begin(), ops.end(), [](KVTuple a, const KVTuple b) {
      return (a.pupc < b.pupc);
    });
  
  for (KVTuple t : ops) {
    assert(t.pupc == (t.v)->pupc);     // Consistency check.

    fprintf(resultsFile, "%-10.llu ", (t.v)->weight);   // Print the dynamic weight.
    fprintf(resultsFile, "%-16.16" PRIx64 " ", t.pupc); // Print the virtual PC.
    print_predicates(resultsFile, (t.v)->PRDst);        // Predicate dests.
    print_cc(resultsFile, (t.v)->CCDst);
    for (int d = 0; d < (t.v)->numGPRDstPtrs; d++)      // GPR dests.
      fprintf(resultsFile, "R%d ", (t.v)->GPRDsts[d]);
    // Print the opcode.
    fprintf(resultsFile, "%s ", SASSIInstrOpcodeStrings[(t.v)->opcode]);    
    print_predicates(resultsFile, (t.v)->PRSrc);        // Predicate srcs.
    print_cc(resultsFile, (t.v)->CCSrc);
    for (int s = 0; s < (t.v)->numGPRSrcPtrs; s++)      // GPR srcs.
      fprintf(resultsFile, "R%d ", (t.v)->GPRSrcs[s]);
    fprintf(resultsFile, "\n");
  }

  fclose(resultsFile);
}


///////////////////////////////////////////////////////////////////////////////////
///
///  Lazily allocate a dictionary before the first kernel launch.
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator mapAllocator([]() {
    sassi_stats = new sassi::dictionary<uint64_t, SASSIInfo>();
  }, sassi_finalize);


///////////////////////////////////////////////////////////////////////////////////
///
///  Records static information about each instruction.  Also records the 
///  dynamic weight associated with each instruction.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_before_handler(SASSIBeforeParams* bp, SASSIRegisterParams *rp)
{
  int threadIdxInWarp = get_laneid();
  int active = __ballot(1);
  int firstActiveThread = (__ffs(active)-1); /*leader*/

  // The warp leader gets to write results.
  if (threadIdxInWarp == firstActiveThread) { 
    // Get the "probably unique" PC.
    uint64_t pupc = bp->GetPUPC();

    // When an instruction is allocated a spot in the hashmap, we will go
    // ahead and initialize the slot with this instruction's static information.
    SASSIInfo *stats = sassi_stats->getOrInit(pupc, [bp,rp,pupc](SASSIInfo *v) {
	v->opcode = bp->GetOpcode();
	v->numGPRDstPtrs = rp->numGPRDstPtrs;
	v->numGPRSrcPtrs = rp->numGPRSrcPtrs;
	v->CCDst = rp->IsCCDefined();
	v->CCSrc = rp->IsCCUsed();
	v->PRDst = rp->GetPredicateDstMask();
	v->PRSrc = rp->GetPredicateSrcMask();
	v->pupc = pupc;
	v->weight = 0;
	for (unsigned d = 0; d < rp->numGPRDstPtrs; d++) 
	  v->GPRDsts[d] = rp->GetRegNum(rp->GetGPRDst(d));
	for (unsigned s = 0; s < rp->numGPRSrcPtrs; s++) 
	  v->GPRSrcs[s] = rp->GetRegNum(rp->GetGPRSrc(s));
      });

    // Every time this instruction is executed, let's increment its weight.
    atomicAdd(&(stats->weight), 1);
  }
}
