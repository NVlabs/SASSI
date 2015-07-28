###############################################################
#
#  Build and install the SASSI libraries.
#
###############################################################

# Setup your variables in env.mk before building!
include ../env.mk

# Decide which libraries you want to compile.
export LIBS := 	libbranch.a \
		libmemdiverge.a \
		libophist.a \
		libopcode.a \
		libstubs.a \
		libvalueprof.a


ifndef INSTALL_DIR
	INSTALL_DIR := ../
endif

NVCC = $(CUDA_HOME)/bin/nvcc
CUPTI = $(CUDA_HOME)/extras/CUPTI

INSTALLED_LIBS := $(addprefix $(INSTALL_DIR)/lib/, $(LIBS))

install: $(INSTALLED_LIBS)

$(INSTALL_DIR)/lib/lib%.a: %.o
	ar r $@ $^
	ranlib $@

%.o: %.cu
	$(NVCC) -ccbin $(CCBIN) -std=c++11 \
		--compiler-options -Wall   \
		$(DEBUG)                   \
		-O3 $(GENCODE)             \
	        -lineinfo -c -rdc=true -o $@ $^   \
		--maxrregcount=16          \
		--compiler-options -Wall   \
		-I$(SASSI_HOME)/include    \
		-I$(CUPTI)/include/        \
		-I$(INSTALL_DIR)/utils/    \
		-I$(INSTALL_DIR)/include

clean:
	$(RM) *.o *.a *~
	$(RM) $(INSTALLED_LIBS)
