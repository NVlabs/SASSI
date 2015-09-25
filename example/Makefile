TARGET := matrixMul
OBJECTS := matrixMul.o

# Copy the environment from env.mk
include ../instlibs/env.mk

# Make sure NVCC uses the C++ compiler we want it to.
export PATH := $(CCBIN):$(PATH)

LINK_FLAGS += -lcudadevrt
NVCC_FLAGS += -g -O3 -dc
CXX_FLAGS += -g -O3

# What example do you want to build?
EXAMPLES := branch none memdiverge opcode ophist valueprof

# Point to SASSI's copy of nvcc.
NVCC := $(SASSI_HOME)/bin/nvcc

# Definitions for the insertion sites.
AFTER_REG := -Xptxas --sassi-inst-after="reg-writes" 
BEFORE_ALL := -Xptxas --sassi-inst-before="all"
BEFORE_COND_BRANCHES := -Xptxas --sassi-inst-before="cond-branches"
BEFORE_MEM := -Xptxas --sassi-inst-before="memory"
BEFORE_REGS := -Xptxas --sassi-inst-before="reg-writes,reg-reads"

# Definitions for the arguments to pass to handlers.
AFTER_REG_INFO := -Xptxas --sassi-after-args="reg-info"
BEFORE_COND_BRANCH_INFO := -Xptxas --sassi-before-args="cond-branch-info"
BEFORE_MEM_INFO := -Xptxas --sassi-before-args="mem-info"
BEFORE_REG_INFO := -Xptxas --sassi-before-args="reg-info"

# If we want to skip predicated off instructions, include this option.
IFF := -Xptxas --sassi-iff-true-predicate-handler-call

DEBUG := -Xptxas --sassi-debug

# We rely heavily on CUPTI to let us know when kernels are launched.
CUPTI_LIB_DIR = $(SASSI_HOME)/extras/CUPTI/lib64
CUPTI = -L$(CUPTI_LIB_DIR) -lcupti 
export LD_LIBRARY_PATH := $(CUPTI_LIB_DIR):$(LD_LIBRARY_PATH)

# The location of our instrumentation libraries.
INST_LIB_DIR = ../instlibs/lib

# Depending on the experiment, let's choose different SASSI options.
branch: EXTRA_NVCC_FLAGS = $(BEFORE_COND_BRANCHES) $(BEFORE_COND_BRANCH_INFO) $(NVCC_FLAGS) 
branch: EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lbranch $(CUPTI) $(LINK_FLAGS) 

memdiverge: EXTRA_NVCC_FLAGS = $(BEFORE_MEM) $(BEFORE_MEM_INFO) $(NVCC_FLAGS) 
memdiverge: EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lmemdiverge $(CUPTI) $(LINK_FLAGS) 

none: EXTRA_NVCC_FLAGS = $(NVCC_FLAGS)
none: EXTRA_LINK_FLAGS = $(LINK_FLAGS) 

opcode: EXTRA_NVCC_FLAGS = $(BEFORE_ALL) $(BEFORE_REG_INFO)  $(NVCC_FLAGS) 
opcode: EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lopcode $(CUPTI) $(LINK_FLAGS) 

ophist: EXTRA_NVCC_FLAGS = $(BEFORE_ALL) $(NVCC_FLAGS) 
ophist: EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lophist $(CUPTI) $(LINK_FLAGS) 

valueprof: EXTRA_NVCC_FLAGS = $(AFTER_REG) $(AFTER_REG_INFO) $(IFF) $(NVCC_FLAGS)
valueprof: EXTRA_LINK_FLAGS = -L$(INST_LIB_DIR) -lvalueprof $(CUPTI) $(LINK_FLAGS) 

$(EXAMPLES): $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) -o $@ $^ $(GENCODE) $(EXTRA_LINK_FLAGS)

%.o:%.cu
	$(NVCC) -I./inc  -c $(GENCODE) $(EXTRA_NVCC_FLAGS) -o $@ $^ 

%.o:%.cpp
	$(NVCC) -I./inc  -c $(CXX_FLAGS) -o $@ $^ 

run: $(TARGET)
	./$(TARGET)

clean: 
	$(RM) -f $(TARGET) *.o

