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

 This is a wrapper around the CUPTI library. It should eventually supplant
 the sassi_lazyallocator.

\****************************************************************************/

#ifndef __SASSI_CUPTIWRAPPER_HPP___
#define __SASSI_CUPTIWRAPPER_HPP___

#include <cupti.h>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include "sassi_intrinsics.h"

namespace sassi {

  class cupti_wrapper {
  public:

    typedef void (*cupti_cb_type)(cupti_wrapper*, const CUpti_CallbackData*);
    
    enum class event_type {
        MODULE_LOAD,      // The driver is loading code onto the device.
	MODULE_UNLOAD,    // The driver is unloading code from the device.
        DEVICE_RESET,     // Device state is about to die.
	KERNEL_LAUNCH     // Kernel enter or exit from launch.
    };

    static const bool callback_before = true;
    static const bool callback_after = false;
    
    /////////////////////////////////////////////////////////////////////////////
    //
    //  Create a new cupti_wrapper, which essentially just creates a managed
    //  CUPTI subscriber.
    //
    /////////////////////////////////////////////////////////////////////////////    
    cupti_wrapper()
    {
      CHECK_CUPTI_ERROR(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)cupti_cb, this),
			"cuptiSubscribe");
    }

    /////////////////////////////////////////////////////////////////////////////
    //
    //  Destroy our cupti_wrapper by unsubscribing our CUPTI subscriber.
    //
    /////////////////////////////////////////////////////////////////////////////    
    ~cupti_wrapper()
    {
      CHECK_CUPTI_ERROR(cuptiUnsubscribe(subscriber), "cuptiUnsubscribe");
    }

    /////////////////////////////////////////////////////////////////////////////
    //
    //  Register for another event type with a new callback function.
    //
    /////////////////////////////////////////////////////////////////////////////    
    void register_callback(event_type cbid, bool before, cupti_cb_type callback) 
    {
      register_callback_h(cbid, before, callback);
    }

  protected:

    struct callback_st {
      bool callBefore;
      cupti_cb_type callbackFn;
    };

    void assign(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, callback_st cbs)
    {
      uCallbackMap.insert(std::make_pair(cbid, cbs));
      CHECK_CUPTI_ERROR(cuptiEnableCallback(true, subscriber, domain, cbid),
			"Problem enabling callback");
    }

    void register_callback_h(event_type cbid, bool before, cupti_cb_type callback)
    {
      callback_st cbs = { before, callback };
      switch (cbid) {
      case event_type::MODULE_LOAD:
	assign(CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED, cbs);
	break;
      case event_type::MODULE_UNLOAD:
	assign(CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING, cbs);
	break;
      case event_type::DEVICE_RESET:
	assign(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020, cbs);
	assign(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020, cbs);
	break;
      case event_type::KERNEL_LAUNCH:
	assign(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020, cbs);
	break;
      }
    }

  protected:

    typedef std::multimap<CUpti_CallbackId,callback_st> event_map;

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
      cupti_wrapper *t = (cupti_wrapper*)userdata;
      event_map::const_iterator lb = t->uCallbackMap.lower_bound(cbid);
      event_map::const_iterator ub = t->uCallbackMap.upper_bound(cbid);

      while (lb != ub)
      {
	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
	  if (((cbInfo->callbackSite == CUPTI_API_ENTER) && lb->second.callBefore) ||
	      ((cbInfo->callbackSite == CUPTI_API_EXIT) && !(lb->second.callBefore))) {
	    lb->second.callbackFn(t, cbInfo);
	  }
	} else if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
	  lb->second.callbackFn(t, cbInfo);
	}
	++lb;
      }
    }

  private:

    CUpti_SubscriberHandle  subscriber;
    event_map               uCallbackMap;
  };

} // End namespace.

#endif
