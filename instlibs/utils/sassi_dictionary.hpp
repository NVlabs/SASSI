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

 A dictionary implementation.
\****************************************************************************/

#ifndef __SASSI_DICTIONARY_HPP___
#define __SASSI_DICTIONARY_HPP___

#include <functional>
#include <nvfunctional>
#include "sassi_managed.hpp"


#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif


namespace sassi {

  template <typename K, typename V>
  class dictionary: public managed
  {
  public:

    typedef unsigned               size_type;
    typedef dictionary<K, V>       self_type;

  public:

    /////////////////////////////////////////////////////////////////////////////
    //
    // Create a new dictionary.  The dictionary is created on the host, and
    // uses UVM (cudaMallocManaged) so that we can view the hashtable on the
    // host after our kernels finish executing.
    //
    /////////////////////////////////////////////////////////////////////////////
    dictionary(size_type slots = 1088723, unsigned maxRetries = 8): 
      m_size(0), m_slots(slots), m_maxRetries(maxRetries)
    {
      cudaMallocManaged(&m_keytable, sizeof(keytable) * slots);
      cudaMallocManaged(&m_valtable, sizeof(V) * slots);
      bzero(m_keytable, sizeof(keytable) * slots);
      bzero(m_valtable, sizeof(V) * slots);
    }
    
    /////////////////////////////////////////////////////////////////////////////
    //
    //  Gets a pointer to the value associated with the key.  If there is no
    //  such key in the dictionary, a key-value mapping is created, and the
    //  "init" function is called to initialize the value.
    //
    /////////////////////////////////////////////////////////////////////////////
    __device__ V *getOrInit(K key, nvstd::function<void (V*)> init)
    {
      V *retVal = NULL;

      uint64_t h1 = key * 1076279;
      uint64_t h2 = (key * 99809343) + 1;
      unsigned attempts = 1;

      // We will try several times to find an empty slot for the key-value
      // pair.  On each try, there are three cases we need to consider.
      for (; attempts <= m_maxRetries; attempts++)
      {
	unsigned h = (unsigned)((h1 + attempts * h2) % m_slots);
	int32_t *metaPtr = &(m_keytable[h].metadata);
	int32_t old = atomicCAS(metaPtr, 0, LOCKED_ENTRY);

	if (old == 0) {
	  // 1. The entry is empty and we got the lock.
	  // Let's initialize the key and call the user's init function.
	  m_keytable[h].key = key;
	  init(&(m_valtable[h]));
	  *metaPtr = VALID_ENTRY;
	  atomicAdd(&m_size, 1);
	  __threadfence();

	  retVal = &(m_valtable[h]);
	  break;
	}

	if (old == LOCKED_ENTRY) {
	  // 2. The entry is locked.  We need to wait for it to be valid.
	  //    We are guaranteed that all other threads in this warp
	  //    would have already unlocked their entries.
	  do {
	    __threadfence();
	    old = *metaPtr;
	  } while (old == LOCKED_ENTRY);
	}

	if (old == VALID_ENTRY) {
	  // 3. The entry is valid according to the metadata.  In this case,
	  //    we need to compare the key to that in the table.  If it is the 
	  //    same, then we can return the value associated with the tuple.
	  //    Otherwise, we need to rehash.
	  bool compares = (key == m_keytable[h].key);
	  if (compares) {
	    retVal = &(m_valtable[h]);
	    break;
	  }
	}	
      }

      assert(attempts < m_maxRetries);
      return retVal;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    //
    //  Return the number of valid entries.
    //
    /////////////////////////////////////////////////////////////////////////////
    CUDA_CALLABLE size_type size() const { return m_size; }

    /////////////////////////////////////////////////////////////////////////////
    //
    //  Returns true if the array is empty, false otherwise.
    //
    /////////////////////////////////////////////////////////////////////////////
    CUDA_CALLABLE bool empty() const { return m_size == 0; }

    /////////////////////////////////////////////////////////////////////////////
    //
    //  Applies 'fun' to every key and value in the dictionary.
    //
    /////////////////////////////////////////////////////////////////////////////
    void map(std::function<void (K&,V&)> fun)
    {
      for (unsigned entry = 0; entry < m_slots; entry++) {
	if (m_keytable[entry].metadata == VALID_ENTRY) {
	  fun(m_keytable[entry].key, m_valtable[entry]);
	}
      }
    }
  
    /////////////////////////////////////////////////////////////////////////////
    //
    //  Empties the dictionary.
    //
    /////////////////////////////////////////////////////////////////////////////
    void clear()
    {
      bzero(m_keytable, sizeof(keytable) * m_slots);
      bzero(m_valtable, sizeof(V) * m_slots);
      m_size = 0;
    }
  
  protected:
  
    /////////////////////////////////////////////////////////////////////////////
    //
    //  This host-side destructor frees up the dynamically allocated memory
    //  on the device.
    //
    /////////////////////////////////////////////////////////////////////////////
    virtual ~dictionary()
    {
      cudaFree(m_keytable);
      cudaFree(m_valtable);
    }
  
  protected:

    struct keytable {
      int32_t metadata;
      K key;
    };

    const int32_t UNLOCKED_ENTRY = 0x0;
    const int32_t VALID_ENTRY    = 0x1;
    const int32_t LOCKED_ENTRY   = 0X2;
  
    size_type m_size;
    const unsigned m_slots;
    const unsigned m_maxRetries;

    keytable *m_keytable;
    V        *m_valtable;
  };
} // End namespace.

#endif /* __SASSI_DICTIONARY_HPP__ */
