
include ../../env.mk

NVCC := $(CUDA_HOME)/bin/nvcc

all: test_dictionary

# Notice we are testing with maxrregcount set at 16.  This is important for SASSI.
# We also need CUPTI for the lazy allocator that we are testing.
test_dictionary: test_dictionary.cu ../sassi_dictionary.hpp ../sassi_managed.hpp ../sassi_lazyallocator.hpp
	$(NVCC) --compiler-options -Wall \
		-O3 -lineinfo -ccbin $(CCBIN) -I$(CUDA_HOME)/extras/CUPTI/include \
		-std=c++11 -maxrregcount=16 -rdc=true \
		$(GENCODE) -o $@ test_dictionary.cu -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti 

clean:
	$(RM) *~ test_dictionary
