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
 * This example computes, for each instruction, for each destination operand,
 * for each bit in the operand, whether or not the bit is constant over all
 * threads and throughout the course of execution of the program.  It also
 * tracks whether an operand is scalar.  
 *
 * The example is based on case study III in the paper,
 *
 *   "Flexible Software Profiling of GPU Architectures"
 *   Stephenson et al., ISCA 2015.
 *
 * The application code the user instruments should be instrumented with the
 * following SASSI flag: -Xptxas --sassi-inst-after="reg-writes"
 *                       -Xptxas --sassi-after-args="reg-info"
 *            [optional] -Xptxas --sassi-iff-true-predicate-handler-call
 *  
\***********************************************************************************/

#define __STDC_FORMAT_MACROS
#include <algorithm>
#include <assert.h>
#include <cupti.h>
#include <inttypes.h>
#include <list>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include "sassi_intrinsics.h"
#include "sassi_dictionary.hpp"
#include "sassi_lazyallocator.hpp"
#include <sassi/sassi-core.hpp>
#include <sassi/sassi-regs.hpp>


///////////////////////////////////////////////////////////////////////////////////
///  
///  Each SASS operation has a number of DSTOperands.  This class
///  keeps track of the stats for a given operand.
///
///////////////////////////////////////////////////////////////////////////////////
class DSTOperand {
public:
  ///////////////////////////////////////////////////////////////////
  ///  
  ///  Initialize the counters for this operand.
  ///
  ///////////////////////////////////////////////////////////////////
  static __device__ void init(DSTOperand *op)
  {
    op->isScalar = -1;
    op->constantOnes = -1;
    op->constantZeros = -1;
  }

  ///////////////////////////////////////////////////////////////////
  ///  
  ///  Prints the bitvector to the given file.
  ///
  ///////////////////////////////////////////////////////////////////
  void print_bits(FILE *f) const
  {
    for (int bn = 31; bn >= 0; bn--) {
      int oneBit = ((constantOnes >> bn) & 0x1);
      int zeroBit = ((constantZeros >> bn) & 0x1);
      if (oneBit == 0 && zeroBit == 0) {
	fprintf(f, "T");
      }
      else if (oneBit == 0 && zeroBit == 1) {
	fprintf(f, "0"); 
      }
      else if (oneBit == 1 && zeroBit == 0) {
	fprintf(f, "1"); 
      }
      else if (oneBit == 1 && zeroBit == 1) {
	fprintf(f, "X");
      }
    }
  }

  ///////////////////////////////////////////////////////////////////
  ///  
  ///  Prints the operand stats to the given file.
  ///
  ///////////////////////////////////////////////////////////////////
  void print(FILE *f)
  {
    fprintf(f, "[%d, \"%s\", %s, [", 
	    regNum, 
	    SASSITypeAsString[regType],
	    isScalar ? "SCALAR " : "VARIANT");
    print_bits(f);
    fprintf(f, "]]");
  }
  
  int regNum;
  int isScalar;
  SASSIType regType;
  int constantOnes;
  int constantZeros;
};


///////////////////////////////////////////////////////////////////////////////////
///  
///  Keep the statistics for each SASS operation.
///
///////////////////////////////////////////////////////////////////////////////////
class SASSOp {
public:
  // This is a bit dirty, and deserves some explanation.  Until
  // device-side allocation of memory matures, we are going to
  // pre-allocate space for the SASSOp statistics on the host.
  // In so doing, we will need to account for the worst-case
  // SASS instruction with regard to number of destination 
  // operands. 
#define MAX_DST_OPERANDS 4

  ///////////////////////////////////////////////////////////////////
  ///  
  ///  Initialize the SASSOp passed in.
  ///
  ///////////////////////////////////////////////////////////////////
  __device__ static void init(SASSOp *op, SASSIRegisterParams *rp)
  {
    op->weight = 0;
    op->numDsts = rp->GetNumGPRDsts();
    assert(op->numDsts <= MAX_DST_OPERANDS);

    // Initialize all of the fields appropriately.
    for (int i = 0; i < op->numDsts; i++) {
      DSTOperand::init(&(op->operands[i]));
      SASSIRegisterParams::GPRRegInfo regInfo = rp->GetGPRDst(i);
      op->operands[i].regNum = rp->GetRegNum(regInfo);
      op->operands[i].regType = rp->GetRegType(regInfo);
    }
  }
  
  ///////////////////////////////////////////////////////////////////
  ///  
  ///  Prints the operation stats to the given file.
  ///
  ///////////////////////////////////////////////////////////////////
  void print(FILE *f)
  {
    fprintf(f, "%lld, [", weight);
    for (int i = 0; i < numDsts; i++) {
      operands[i].print(f);
    }
    fprintf(f, "]");
  }
  
  unsigned long long weight;
  int numDsts;
  DSTOperand operands[MAX_DST_OPERANDS];
};


/// The actual dictionary, declared as a UVM managed type so we can access it on 
/// the device and host.
static __managed__ sassi::dictionary<uint64_t, SASSOp> *sassi_stats;


///////////////////////////////////////////////////////////////////////////////////
///  
///  We will register this function to be called whenever the device is reset, 
///  or when the program is about to exit.  The function will print out the 
///  aggregated statistics.
///
///////////////////////////////////////////////////////////////////////////////////
static void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason)
{
  struct KVTuple {
    uint64_t k;
    SASSOp *v;
  };

  FILE *resultsFile = fopen("sassi-valueprof.txt", "w");
  
  fprintf(resultsFile, "\nValue profiling results\n");
  fprintf(resultsFile, "ADDRESS | WEIGHT | [regnum, type, scalarness, bitstring]*\n");
  fprintf(resultsFile, "---------------------------------------------------------\n");
  
  std::vector<KVTuple> ops;
  sassi_stats->map([&ops](uint64_t& key, SASSOp &val) {
      ops.push_back({key, &val});
    });
  
  std::sort(ops.begin(), ops.end(), [](KVTuple a, const KVTuple b) {
      return a.k < b.k;
    });
  
  for (KVTuple t : ops) {
    fprintf(resultsFile, "[%.16" PRIx64 ", ", t.k);
    t.v->print(resultsFile);
    fprintf(resultsFile, "]\n");
  }
  
  cudaDeviceSynchronize();
  fclose(resultsFile);
}

///////////////////////////////////////////////////////////////////////////////////
///
///  Lazily allocate a dictionary before the first kernel launch.
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator mapAllocator([]() {
    sassi_stats = new sassi::dictionary<uint64_t, SASSOp>();
  }, sassi_finalize);


///////////////////////////////////////////////////////////////////////////////////
//
//  This example uses the atomic bitwise operations to keep track of the constant
//  bits produced by each instruction.
//
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_after_handler(SASSIAfterParams* ap, SASSIRegisterParams *rp)
{
  int threadIdxInWarp = get_laneid();
  int firstActiveThread = (__ffs(__ballot(1))-1); /*leader*/

  // Get the "probably unique" PC.
  uint64_t pupc = ap->GetPUPC();

  // The dictionary will return the SASSOp associated with this PC, or insert
  // it if it does not exist.  If it does not exist, the lambda passed as
  // the second argument to getOrInit is used to initialize the SASSOp.
  SASSOp *stats = sassi_stats->getOrInit(pupc, [&rp](SASSOp *v) {
      SASSOp::init(v, rp);
    });
  
  // Record the number of times the instruction executes.
  atomicAdd(&(stats->weight), 1);
  for (int d = 0; d < rp->GetNumGPRDsts(); d++) {
    // Get the value in each destination register.
    SASSIRegisterParams::GPRRegInfo regInfo = rp->GetGPRDst(d);
    SASSIRegisterParams::GPRRegValue regVal = rp->GetRegValue(ap, regInfo); 

    // Use atomic AND operations to track constant bits.
    atomicAnd(&(stats->operands[d].constantOnes), regVal.asInt); 
    atomicAnd(&(stats->operands[d].constantZeros), ~regVal.asInt);

    int leaderValue = __shfl(regVal.asInt, firstActiveThread); 
    int allSame = (__all(regVal.asInt == leaderValue) != 0);
    // The warp leader gets to write results.
    if (threadIdxInWarp == firstActiveThread) { 
      atomicAnd(&(stats->operands[d].isScalar), allSame);
    }
  }
}




