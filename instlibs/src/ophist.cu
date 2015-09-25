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
 * This example shows how to use SASSI to create a histogram of opcodes
 * encountered during the execution of a program.  Unlike many of the other
 * examples we include, this example does not use Unified Virtual Memory (UVM),
 * and is intended as an example that applies across all NVIDIA's architectures
 * from Fermi to Maxwell.
 *
 * The application code the user instruments should be instrumented with the
 * following SASSI flag: "-Xptxas --sassi-inst-before=all".
 *  
\***********************************************************************************/

#include <cupti.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include "sassi_intrinsics.h"
#include "sassi_lazyallocator.hpp"
#include <sassi/sassi-core.hpp>

// Keep track of all the opcodes that were executed.  Normally, we would declare
// this array to be __managed__ so that UVM would take care of copying data
// back and forth.  In this example, we will just declare this array to reside
// on the device, and we will explicitly copy the data back and forth.
__device__ unsigned long long dynamic_instr_counts[SASSI_NUM_OPCODES];

///////////////////////////////////////////////////////////////////////////////////
///
///  This is a SASSI handler that handles only basic information about each
///  instrumented instruction.  The calls to this handler are placed by
///  convention *before* each instrumented instruction.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_before_handler(SASSIBeforeParams* bp)
{
    if (bp->GetInstrWillExecute())
    {
      SASSIInstrOpcode op = bp->GetOpcode();
      atomicAdd(&(dynamic_instr_counts[op]), 1ULL);
    }
}

///////////////////////////////////////////////////////////////////////////////////
/// 
///  Write out the statistics we've gathered.
/// 
///////////////////////////////////////////////////////////////////////////////////
static void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason)
{
  unsigned long long instr_counts[SASSI_NUM_OPCODES];

  // Copy the data off of the device.
  CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&instr_counts, dynamic_instr_counts, sizeof(instr_counts)));

  FILE *resultFile = fopen("sassi-ophist.txt", "w");
  for (unsigned i = 0; i < SASSI_NUM_OPCODES; i++) {
    if (instr_counts[i] > 0) {
      fprintf(resultFile, "%-10.10s: %llu\n", SASSIInstrOpcodeStrings[i], instr_counts[i]);
    }
  }
  fclose(resultFile);
}

///////////////////////////////////////////////////////////////////////////////////
///
///  Lazily initialize the counters before the first kernel launch. 
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator counterInitializer(
  /* Initialize the counters. */
  []() {
    unsigned long long instr_counts[SASSI_NUM_OPCODES];
    bzero(instr_counts, sizeof(instr_counts));
    // Initialize the array we allocated on the device.
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dynamic_instr_counts, &instr_counts, sizeof(instr_counts)));
  }, sassi_finalize);

