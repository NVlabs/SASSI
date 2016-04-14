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
 * This header file provides support for mapping SASSI's PUPC (program counter)
 * to source code lines.  The class, src_mapper, derives from cupti_wrapper, 
 * which allows us to intercept CUDA runtime events.  In particular, src_mapper
 * tracks "module loads", "kernel launches", and "device resets".  When the 
 * module is loaded, src_mapper gets the "cubin" (the device-side object file),
 * disassembles it via "nvdisasm", and then parses the result to get source
 * mappings.
 *
 * Dependencies: 1) Be sure have "nvdisasm" in your path.  This is often in
 *                  /usr/local/cuda7/bin or you can use the version in your
 *                  SASSI install location.
 *               2) OpenSSL crypto library (libcrypto): To get a hash of the 
 *                  function name, which SASSI uses to generate PUPCs.
 *               3) Boost regex: To parse the disassembled cubin.
 *
 *               Be sure to link with -lcrypto -lboost_regex
 *  
\***********************************************************************************/

#ifndef __SASSI_SRCMAP_HPP___
#define __SASSI_SRCMAP_HPP___

#include <fstream>
#include <cupti.h>
#include <iomanip>
#include <iterator>
#include <map>
#include <string>
#include <sstream>
#include <boost/regex.hpp>
#include <openssl/sha.h>
#include "sassi_cuptiwrapper.hpp"

namespace sassi {

  struct loc_info {
    std::string sass;           // The actual SASS at this location.
    const std::string *file_name;     // A pointer to the filename.
    const std::string *function_name; // A pointer to the function name.
    unsigned line_num;          // The line of source.
  };

  class src_mapper : public cupti_wrapper {
  public:

    ////////////////////////////////////////////////////////////////////////////////
    //
    //  Create a src_mapper, whose function is to map a SASSI PUPC to a line of
    //  CUDA source.  As usual to have access to line information embedded with
    //  the code, you must use the "-lineinfo" option.
    //
    ////////////////////////////////////////////////////////////////////////////////    
    src_mapper(): device_state_valid(false)
    {
      register_callback(event_type::MODULE_LOAD, callback_after, dump_pc_mapping);
      register_callback(event_type::KERNEL_LAUNCH, callback_before, set_valid_data);
      register_callback(event_type::DEVICE_RESET, callback_after, reset_valid_data);
    }

    ////////////////////////////////////////////////////////////////////////////////    
    //
    // Gets a hashmap that maps from PUPCs to locations.
    //
    //////////////////////////////////////////////////////////////////////////////// 
    const std::map<uint64_t, loc_info>& get_location_map() const { return location_map; }

    ////////////////////////////////////////////////////////////////////////////////    
    //
    // The source mapper keeps track of whether state you have stored in device
    // memory should be valid.  If the return is true, your device-side state
    // (e.g., SASSI counters) should be valid; otherwise, your device-side state
    // is probably garbage.
    //
    //////////////////////////////////////////////////////////////////////////////// 
    bool is_device_state_valid() const { return device_state_valid; }

  protected:

    ////////////////////////////////////////////////////////////////////////////////
    //
    //  Uses CUPTI's facilities to dump the CUBIN associated with this module.
    //
    ////////////////////////////////////////////////////////////////////////////////    
    static std::string dump_cumodule(void *resourceDescriptor)
    {
      const char *pCubin;
      size_t cubinSize;
      
      CUpti_ModuleResourceData *moduleResourceData = 
	(CUpti_ModuleResourceData *)resourceDescriptor; 
      
      std::string cubinFileName = tmpnam(nullptr);
      pCubin = moduleResourceData->pCubin;
      cubinSize = moduleResourceData->cubinSize;
      
      FILE *cubin;
      cubin = fopen(cubinFileName.c_str(), "wb");
      fwrite(pCubin, sizeof(uint8_t), cubinSize, cubin);
      fclose(cubin);
      return cubinFileName;
    }


    ////////////////////////////////////////////////////////////////////////////////
    //
    //  Uses off-the-shelf NVIDIA commands to disassemble the CUBIN and dump it
    //  to a file.  Make sure that nvdisasm is in your path.  It will be in the
    //  same location as nvcc, ptxas, etc.
    //
    ////////////////////////////////////////////////////////////////////////////////    
    static std::string create_sass(const std::string fname)
    {
      std::string sfname = fname + ".sass";
      std::string command = "nvdisasm -g -ndf -c " + fname + " > " + sfname;
      
      int status = system(command.c_str());
      if (status != 0) {
	fprintf(stderr, "::: ERROR: Make sure nvdisasm is in your PATH :::\n");
	assert(0);
      }
      
      return sfname;
    }


    ////////////////////////////////////////////////////////////////////////////////
    //
    //  Parses the dumped file to get the SASS correlations.
    //
    ////////////////////////////////////////////////////////////////////////////////    
    void get_sass_corrs(std::string sass_filename)
    {
      std::ifstream ifs(sass_filename.c_str());
      boost::regex funNameRegex("\\.text\\.(\\w+):");
      boost::regex fileNameRegex("\\s*\\/\\/## File \"([^\"]+)\", line (\\d+)$");
      boost::regex SASSLineRegex("\\s*\\/\\*([0-9a-f]+)\\*\\/\\s+([^$]+)$");
      
      uint64_t addrBase = 0;
      std::string functionName;
      std::string fileName = "Location not known";
      std::string lineNumber = "-1";
      std::string line;
      
      while (getline(ifs, line))
	{
	  // Check for .text.<name>: and functionName <- <name>
	  // Compute SHA256 hash of functionName
	  boost::cmatch match;
	  if (boost::regex_match(line.c_str(), match, funNameRegex))
	    {
	      unsigned char md[SHA_DIGEST_LENGTH];
	      functionName = std::string(match[1].first, match[1].second);
	      if (SHA1((const unsigned char*)functionName.c_str(), 
		       functionName.length(), md)) {
		void *mdv = (void*)md;
		uint32_t v = *((uint32_t*)mdv);
		addrBase = ((uint64_t)v << 32);
	      } else {
		fprintf(stderr, "::: ERROR: During SHA1 hashing :::\n");
		assert(0);
	      }
	    }
	  
	  // Get the file name and line number.
	  if (boost::regex_match(line.c_str(), match, fileNameRegex))
	    {
	      fileName = std::string(match[1].first, match[1].second);
	      lineNumber = std::string(match[2].first, match[2].second);
	    }

	  // Gets the canonical file and function name.
	  std::pair<std::set<std::string>::iterator,bool> file_it = 
	    file_names.insert(fileName);;
	  std::pair<std::set<std::string>::iterator,bool> function_it = 
	    function_names.insert(functionName);;
	  
	  // Look for /*[a-f0-9]+*/, put the hex value in offset
	  // Also, extract the SASS, which is everything after this match
	  if (boost::regex_match(line.c_str(), match, SASSLineRegex))
	    {
	      loc_info info;
	      std::stringstream offsetStream;
	      uint64_t offset, pupc;

	      // Compute the PUPC using hash and offset.
	      std::string offsetString = std::string(match[1].first, match[1].second);
	      offsetStream << std::hex << offsetString;
	      offsetStream >> offset;
	      pupc = addrBase | offset;
	      
	      // Put the loc and SASS in hash map using PUPC as the key.
	      info.sass = std::string(match[2].first, match[2].second);
	      info.file_name = &(*(file_it.first));
	      info.function_name = &(*(function_it.first));
	      info.line_num = std::stoi(lineNumber);
	      location_map.insert(std::pair<uint64_t, loc_info>(pupc, info));
	    }
	}
    }

    ////////////////////////////////////////////////////////////////////////////////
    // 
    //  When the module is loaded, this callback will be called and we can get
    //  the source code mappings.
    //
    ////////////////////////////////////////////////////////////////////////////////
    static void 
    dump_pc_mapping(cupti_wrapper* wrapper, const CUpti_CallbackData *cbdata)
    {
      CUpti_ResourceData *rdata = (CUpti_ResourceData*)cbdata;
      std::string cubinFileName = dump_cumodule(rdata->resourceDescriptor);
      std::string sassFileName = create_sass(cubinFileName);
      ((src_mapper*)wrapper)->get_sass_corrs(sassFileName);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // 
    //  Go ahead and set the device_state_valid flag to true when a kernel is 
    //  launched.
    //
    ////////////////////////////////////////////////////////////////////////////////
    static void 
    set_valid_data(cupti_wrapper* wrapper, const CUpti_CallbackData *cbdata) 
    {
      ((src_mapper*)wrapper)->device_state_valid = true;
    }
	
    ////////////////////////////////////////////////////////////////////////////////
    // 
    //  This callback is fired when the device is reset.  We set the
    //  device_state_valid flag to false. 
    //
    ////////////////////////////////////////////////////////////////////////////////
    static void 
    reset_valid_data(cupti_wrapper* wrapper, const CUpti_CallbackData *cbdata) 
    {
      ((src_mapper*)wrapper)->device_state_valid = false;
    }
    
  protected:

    bool device_state_valid;
    std::set<std::string> file_names;
    std::set<std::string> function_names;
    std::map<uint64_t, loc_info> location_map;
  };

}

#endif
