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
 * This example shows how to use SASSI to inspect the control flow graph.
 *
 * The application code the user instruments should be instrumented with the
 * following SASSI flags: -Xptxas --sassi-function-entry -Xptxas --sassi-bb-entry
 *  
\***********************************************************************************/

#include <algorithm>
#include <assert.h>
#include <cupti.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include "sassi_intrinsics.h"
#include "sassi_lazyallocator.hpp"
#include "sassi_dictionary.hpp"
#include <sassi/sassi-function.hpp>

// 8Mb of space for CFG information.
#define POOLSIZE (1024 * 1024 * 8)
#define MAX_FN_STR_LEN 64

// Create a memory pool that we can populate on the device and read on the host.
static __managed__ uint8_t sassi_mempool[POOLSIZE];
static __managed__ int     sassi_mempool_cur;

// A structure to record a basic block.  We will perform a deep copy
// of SASSI's SASSIBasicBlockParams for each basic block.
struct BLOCK {
  int id;
  unsigned long long weight;
  bool isEntry;
  bool isExit;
  int numInstrs;
  int numSuccs;
  int succs[2];
};

// A structure to record a function's CFG.
struct CFG {
  char fnName[MAX_FN_STR_LEN];
  int numBlocks;
  BLOCK *blocks;
};

// A dictionary of CFGs.
static __managed__ sassi::dictionary<int64_t, CFG*> *sassi_cfg;

// A dictionary of basic blocks.
static __managed__ sassi::dictionary<int64_t, BLOCK*> *sassi_cfg_blocks;

///////////////////////////////////////////////////////////////////////////////////
///
///  Allocate data from the UVM mempool.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void *simple_malloc(size_t sz)
{
  int ptr = atomicAdd(&sassi_mempool_cur, sz);
  assert ((ptr + sz) <= POOLSIZE);
  return (void*) &(sassi_mempool[ptr]);
}

///////////////////////////////////////////////////////////////////////////////////
///
///  A simple string copy to copy from device memory to our UVM malloc'd region.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void simple_strncpy(char *dest, const char *src)
{
  int i;
  for (i = 0; i < MAX_FN_STR_LEN-1; i++) {
    char c = src[i];
    if (c == 0) break;
    dest[i] = c;
  }
  dest[i] = '\0';
}

///////////////////////////////////////////////////////////////////////////////////
///
///  A call to this function will be inserted at the beginning of every 
///  CUDA function or kernel.  We will essentially perform a deep copy of the
///  CFG SASSI presents.  All of the copied data only contains static information
///  about the CFG.  In the sassi_basic_block_entry handler, below, we will 
///  record the dynamic number of times the basic block is invoked.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_function_entry(SASSIFunctionParams* fp)
{
  int numBlocks = fp->GetNumBlocks();
  const SASSIBasicBlockParams * const * blocks = fp->GetBlocks();
  
  CFG *cPtr = *(sassi_cfg->getOrInit((int64_t)fp, [numBlocks, blocks, fp](CFG **cfg) {
      CFG *cPtr = (CFG*) simple_malloc(sizeof(CFG));
      simple_strncpy(cPtr->fnName, fp->GetFnName());
      cPtr->numBlocks = numBlocks;
      cPtr->blocks = (BLOCK*) simple_malloc(sizeof(BLOCK) * numBlocks);
      *cfg = cPtr;
  }));

  __threadfence();

  for (int bb = 0; bb < numBlocks; bb++) {
    const SASSIBasicBlockParams *blockParam = blocks[bb];
    BLOCK *blockPtr = &(cPtr->blocks[bb]);    
    sassi_cfg_blocks->getOrInit((int64_t)blockParam, [blockParam, blockPtr](BLOCK **bpp) {
	*bpp = blockPtr;
	blockPtr->id = blockParam->GetID();
	blockPtr->weight = 0;
	blockPtr->isEntry = blockParam->IsEntryBlock();
	blockPtr->isExit = blockParam->IsExitBlock();
	blockPtr->numInstrs = blockParam->GetNumInstrs();
	blockPtr->numSuccs = blockParam->GetNumSuccs();
	assert(blockParam->GetNumSuccs() <= 2);
	const SASSIBasicBlockParams * const * succs = blockParam->GetSuccs();
	for (int s = 0; s < blockParam->GetNumSuccs(); s++) {
	  blockPtr->succs[s] = succs[s]->GetID();
	}
      });
  }
}

///////////////////////////////////////////////////////////////////////////////////
///
///  Simply lookup the basic block in our dictionary, get its "weight" feild
///  and increment it.
///
///////////////////////////////////////////////////////////////////////////////////
__device__ void sassi_basic_block_entry(SASSIBasicBlockParams *bb)
{
  BLOCK **blockStr = sassi_cfg_blocks->getOrInit((int64_t)bb, [](BLOCK **bpp) { assert(0); });
  atomicAdd(&((*blockStr)->weight), 1);
}

///////////////////////////////////////////////////////////////////////////////////
///
///  Print the graph out in "dot" format.  
///  E.g., use:
///
///       dot -Tps -o graph.ps sassi-cfg.dot 
///
///  to render the graph in postscript.
///
///////////////////////////////////////////////////////////////////////////////////
static void sassi_finalize(sassi::lazy_allocator::device_reset_reason unused)
{
  cudaDeviceSynchronize();
  FILE *cfgFile = fopen("sassi-cfg.dot", "w");
  sassi_cfg->map([cfgFile](int64_t k, CFG* &cfg) {
      fprintf(cfgFile, "digraph %s {\n", cfg->fnName);
      double weightMax = 0.0;
      for (int bb = 0; bb < cfg->numBlocks; bb++) {
	BLOCK *block = &(cfg->blocks[bb]);
	weightMax = std::max(weightMax, (double)block->weight);
      }
      for (int bb = 0; bb < cfg->numBlocks; bb++) {
	BLOCK *block = &(cfg->blocks[bb]);
	int per = block->isExit ? 3 : 1;
	int boxWeight = 100 - std::round(100.0 * ((double)block->weight / weightMax));
	int fontWeight = boxWeight > 40 ? 0 : 100;
	fprintf(cfgFile, "\tBB%d [style=filled,fontcolor=gray%d,shape=box,"
		"peripheries=%d,color=gray%d,label=\"BB%d : %d ins\"];\n", 
		block->id, fontWeight, per, boxWeight, block->id, block->numInstrs);
      }
      for (int bb = 0; bb < cfg->numBlocks; bb++) {
	BLOCK *block = &(cfg->blocks[bb]);
	for (int s = 0; s < block->numSuccs; s++) {
	  fprintf(cfgFile, "\tBB%d -> BB%d;\n", block->id, block->succs[s]);
	}
      }
      fprintf(cfgFile, "}\n");
    });
  fclose(cfgFile);
}

///////////////////////////////////////////////////////////////////////////////////
///
///  Initialize the UVM memory pool and our two dictionaries.  
///
///////////////////////////////////////////////////////////////////////////////////
static void sassi_init()
{
  sassi_mempool_cur = 0;
  bzero(sassi_mempool, sizeof(sassi_mempool));
  sassi_cfg = new sassi::dictionary<int64_t, CFG*>(601);
  sassi_cfg_blocks = new sassi::dictionary<int64_t, BLOCK*>(7919);
}


///////////////////////////////////////////////////////////////////////////////////
///
///  
///
///////////////////////////////////////////////////////////////////////////////////
static sassi::lazy_allocator mapAllocator(sassi_init, sassi_finalize);
