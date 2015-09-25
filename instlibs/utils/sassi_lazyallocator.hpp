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

 This class allows us to lazily call an allocation/initialization
 function once and only once before the first kernel launch.  It uses
 CUPTI to catch thread launches, and on the first thread launch, it 
 calls the callback the user registered with. We do this because for 
 SASSI libraries we generally don't have control over when things are 
 initialized.  If we try to do CUDA or CUPTI related things in a static 
 constructor, it is very possible that the CUDA runtime has not even 
 been initialized.
 This class also provides an interface to call a function before and/or after
 kernel calls. 
\****************************************************************************/

#ifndef __SASSI_LAZYALLOCATOR_HPP___
#define __SASSI_LAZYALLOCATOR_HPP___

#include <cupti.h>
#include <cstdlib>
#include <string>
#include <list>
#include "sassi_intrinsics.h"

namespace sassi {

  /////////////////////////////////////////////////////////////////////////////
  //
  // Lazily calls an allocation/initialization function once before the
  // first kernel launch.
  //
  /////////////////////////////////////////////////////////////////////////////
  class lazy_allocator {
  public:

    enum class device_reset_reason {
      DEVICE_RESET,
      PROGRAM_EXIT
    };

    typedef lazy_allocator  self_type;
    typedef void (*device_init_cb_type)();
    typedef void (*device_reset_cb_type)(device_reset_reason);
    typedef void (*kernel_ee_cb_type)(const CUpti_CallbackData*);
    
    /////////////////////////////////////////////////////////////////////////////
    //
    //  Registers the user's callbacks.
    //
    /////////////////////////////////////////////////////////////////////////////    
    lazy_allocator(device_init_cb_type init_cb,
		   device_reset_cb_type reset_cb, 
		   kernel_ee_cb_type kernel_entry_cb = [](const CUpti_CallbackData*){}, 
		   kernel_ee_cb_type kernel_exit_cb = [](const CUpti_CallbackData*){}):
      init_cb(init_cb), reset_cb(reset_cb), 
      entry_cb(kernel_entry_cb), exit_cb(kernel_exit_cb), 
      valid_data(false)
    {
      CHECK_CUPTI_ERROR(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)cupti_cb, this),
			"cuptiSubscribe");
      setupLaunchCB();
      setupResetCB();
    }
    
    ~lazy_allocator() 
    {
      if (valid_data && reset_cb) {
	reset_cb(device_reset_reason::PROGRAM_EXIT);
      }
    }
    
  protected:    

    /////////////////////////////////////////////////////////////////////////////
    //
    //  Sets up a one-time callback for kernel launches.  If the device is
    //  reset, then this callback will be setup again.
    //
    /////////////////////////////////////////////////////////////////////////////    
    void setupLaunchCB()
    {
      if (init_cb) {
	CHECK_CUPTI_ERROR(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
					      CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020),
			  "Problem enabling callback");
      }
    }
    
    /////////////////////////////////////////////////////////////////////////////
    //
    //  Sets up callbacks for when the device is reset.
    //
    /////////////////////////////////////////////////////////////////////////////
    void setupResetCB()
    {
      if (reset_cb) {
	CHECK_CUPTI_ERROR(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
					      CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020),
			  "Problem enabling callback");
	CHECK_CUPTI_ERROR(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
					      CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020),
			  "Problem enabling callback");
      }
    }

    /////////////////////////////////////////////////////////////////////////////
    //
    //  Our main CUPTI callback for handling launches and cases where the 
    //  device is reset.
    //
    /////////////////////////////////////////////////////////////////////////////
    static void CUPTIAPI cupti_cb(void *userdata,
				  CUpti_CallbackDomain domain,
				  CUpti_CallbackId cbid,
				  const CUpti_CallbackData *cbInfo)
    {
      self_type *ld = (self_type*) userdata;

      if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020) ||
          (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020))
      {
	if (ld->valid_data && ld->reset_cb) {
	  ld->reset_cb(device_reset_reason::DEVICE_RESET);
	  ld->valid_data = false;
	}
      }
      else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020)
      {
        // Call the init_cb function only once. 
        if(!ld->valid_data) {
          ld->valid_data = true;
          ld->init_cb();
        }
	
        if (cbInfo->callbackSite == CUPTI_API_ENTER) { 
	  // Call the user defined entry_cb function on every kernel entry
          ld->entry_cb(cbInfo);
        } else if (cbInfo->callbackSite == CUPTI_API_EXIT) { 
	  // Call the user defined exit_cb function on every kernel exit
          ld->exit_cb(cbInfo);
        }
      }
    }

  private:

    CUpti_SubscriberHandle     subscriber;
    bool                       valid_data;
    const device_init_cb_type  init_cb;
    const device_reset_cb_type reset_cb;
    const kernel_ee_cb_type    entry_cb;
    const kernel_ee_cb_type    exit_cb;
  };

} // End namespace.

#endif
