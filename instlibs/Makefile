
# Setup your variables in env.mk before building!
include env.mk

export PATH:=$(PATH):$(CUDA_HOME)/bin/
export INSTALL_DIR := $(PWD)

all: instlibs

# This target will build all the example libraries,
# but it requires 3.0+ devices.
instlibs: lib
	make -C src install

# Use this target if you targeting a Fermi device.
fermi-only: lib
	make -C src install-fermi-only

lib:
	mkdir lib

clean:
	make -C src clean
	$(RM) -r lib
