###############################################################
#
#  Build and install the SASSI libraries.
#
###############################################################

# Setup your variables in env.mk before building!
include ../env.mk

# These libraries work with Fermi, the others do not.
export FERMI_LIBS := libophist_fermi.a libstubs.a

export PATH := $(CCBIN):$(PATH)

# Decide which libraries you want to compile.
export LIBS := 	libstubs.a \
		libbranch.a \
		libcfg.a \
		libmemdiverge.a \
		libopcode.a \
		libophist.a \
		libvalueprof.a \

ifndef INSTALL_DIR
	INSTALL_DIR := ../
endif

NVCC = $(CUDA_HOME)/bin/nvcc
CUPTI = $(CUDA_HOME)/extras/CUPTI

INSTALLED_LIBS := $(addprefix $(INSTALL_DIR)/lib/, $(LIBS))
INSTALLED_FERMI_LIBS := $(addprefix $(INSTALL_DIR)/lib/, $(FERMI_LIBS))

install: $(INSTALLED_LIBS) $(INSTALLED_FERMI_LIBS)

install-fermi-only: $(INSTALLED_FERMI_LIBS)

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
		-I$(BOOST_HOME)/include    \
		-I$(CUPTI)/include/        \
		-I$(INSTALL_DIR)/utils/    \
		-I$(INSTALL_DIR)/include

clean:
	$(RM) *.o *.a *~ 
	$(RM) $(INSTALLED_LIBS)
