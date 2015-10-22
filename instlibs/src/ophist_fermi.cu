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
 * Additionally, we rely on ld's "--wrap" functionality to "intercept" 
 * main and exit.  SASSI issues #4 and #6 prompted this approach.  The reason
 * we intercept main and exit is commented below.
 *  
\***********************************************************************************/

#include <assert.h>
#include <cupti.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
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
// The file that we will dump results to.  We will initialize this in our
// __wrap_main() function, which will be called before the instrumented program's
// main if we compile the application correctly.
///////////////////////////////////////////////////////////////////////////////////
static FILE *resultFile = NULL;

///////////////////////////////////////////////////////////////////////////////////
/// 
///  Get the counters off the device and dump them.
/// 
///////////////////////////////////////////////////////////////////////////////////
static void collect_and_dump_counters(const char *when)
{
  unsigned long long instr_counts[SASSI_NUM_OPCODES];

  assert(resultFile != NULL);
  
  // Copy the data off of the device.
  CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&instr_counts, dynamic_instr_counts, sizeof(instr_counts)));

  fprintf(resultFile, "----- Results %s -----\n", when);
  for (unsigned i = 0; i < SASSI_NUM_OPCODES; i++) {
    if (instr_counts[i] > 0) {
      fprintf(resultFile, "%-10.10s: %llu\n", SASSIInstrOpcodeStrings[i], instr_counts[i]);
    }
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
  collect_and_dump_counters("before program explicitly ended");
  fclose(resultFile);
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
  resultFile = fopen("sassi-ophist-fermi.txt", "w");

  int ret = __real_main(argc, argv);
  collect_and_dump_counters("before program implicitly ended");
  fclose(resultFile);
  return ret;
}


///////////////////////////////////////////////////////////////////////////////////
/// 
///  We can still use a CUPTI callback to find out when the device has been
///  explictly reset with a call to cudaDeviceReset().  We want to intercept
///  that call and dump our counters before they are obliterated.
/// 
///////////////////////////////////////////////////////////////////////////////////
void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason)
{
  // Unlike in the UVM case, we can't dump the counters at exit because
  // there is no way for us to guarantee that the CUDA runtime is still
  // available.  See issues #4 and #6.
  if (reason == sassi::lazy_allocator::device_reset_reason::DEVICE_RESET) {
    collect_and_dump_counters("before device explicitly reset");
  }
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

