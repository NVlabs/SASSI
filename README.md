=======
News
==========================================

* Releasing a new version of the SASSI binary today (4/14/2015) that
  includes some bug fixes and additional features.

* Including a new header-file library that allows users to correlate
  SASS locations with the corresponding CUDA source line.

* We conducted a [tutorial at
  MICRO-48](https://github.com/NVlabs/SASSI/wiki/MICRO48-Tutorial).
  You can check out the [slide
  deck](https://github.com/NVlabs/SASSI/blob/master/doc/SASSI-Tutorial-Micro2015.pptx)
  that we covered.

SASSI Instrumentation Tool for NVIDIA GPUs
==========================================

This project contains the SASSI instrumentation tool.  SASSI is not
part of the official CUDA toolkit, but instead is a research prototype
from the Architecture Research Group at NVIDIA.

SASSI is a selective instrumentation framework
for NVIDIA GPUs.  SASSI stands for SASS Instrumenter, where SASS is
NVIDIA's name for its native ISA.  SASSI is a pass in NVIDIA's backend
compiler, ptxas, that selectively inserts instrumentation
code.  The purpose of SASSI is to allow users to measure or modify
ptxas-generated SASS by injecting instrumentation code
*during* code generation.

NVIDIA has many excellent development tools. Why the need for another
tool? NVIDIA's tools such as cuda-memcheck and nvvp provide excellent,
but *fixed-function* inspection of programs.  While they are great at
what they are designed for, the user has to choose from a fixed menu
of program characteristics to measure.  If you want to measure some
aspect of program execution outside the purview of those tools you are
out of luck.  SASSI allows users to flexibly inject their own
instrumentation to measure novel aspects of GPGPU execution.

SASSI consists of two main components:
* A closed-source fork of NVIDIA's PTX assembler, ptxas, that is capable of
injecting instrumentation code during compilation.  SASSI's version of
ptxas is distributed on GitHub via "Releases".
* Several realistic samples that demonstrate SASSI's operation.

Newest release notes
==========================================

* We have added some new features. There is a new instrumentation
  library that demonstrates how to map a SASS instruction with a given
  PUPC (SASSI's version of a PC) to the CUDA source.  See the "branch"
  library for its usage.  Also see the `branch` target in
  `example/Makefile` for the compiler flags necessary to use the new
  feature.

* Support for emulating novel SASS instructions for ISA exploration is
  more stable.  We have not yet documented this feature because we are
  still working out the kinks, but if you are interested in this
  feature, please contact me.

* Bug fix.  The PUPC was invalid for functions with long names.  This
  fix requires installing the latest SASSI binaries.

Prerequisites
------------------

SASSI has the following system prerequisites:

1. Platform requirement: SASSI requires an X86 64-bit host; a Fermi-,
  Kepler-, or Maxwell-based GPU; and at the time of this writing we
  have generated SASSI for Ubuntu (12, 14, and 15), Debian 7 and 8, and CentOS 6 and 7.
2. Install CUDA 7: At the time of this writing, CUDA 7 can be
  fetched [from here](https://developer.nvidia.com/cuda-toolkit-70).
3. Make sure you have a 346.41 driver or newer: The CUDA 7
  installation script can install a new driver for you that meets this
  requirement.  If you already have a newer driver, that should be
  fine.  You can test your driver version with the `nvidia-smi`
  command.
4. The installation script requires Python 2.7 or newer.

Installation
------------------

After you have fulfilled your prerequisites, install SASSI by doing the following:

1. Find the release for your platform by clicking on the "release" tab on the
GitHub project page, or by [navigating
here](https://github.com/NVlabs/SASSI/releases). Find your
architecture in the "Downloads" list and download.  This download is a
very simple binary installer.
2. Run the installer via `sh`, for example, `sh SASSI_x86_64_centos_6.run`.

You might need to run the installer as root, depending on where you
plan to install SASSI.

Usage
------------------

For usage, please follow the instructions in the user guide, which you
can find in `doc/sassi-user-guide.pdf`.

Additionally, `ptxas -h` lists SASSI's supported options.

Restrictions and caveats
------------------

1. 32-bit architectures are not supported.

    This was an early design decision to reduce the large cross product of
    possible configurations.  Please let us know if 32-bit support would
    be useful though, because it probably wouldn't be too hard to
    support.

2. Programs currently have to be compiled with `-rdc=true`, which
affects performance.

    SASSI allows users to instrument code by injecting function calls to
    user-defined functions that are later linked in.  In order to perform
    cross-module function calls in CUDA one must use the "relocatable
    device code" option, `-rdc=true`.  Future versions of SASSI may remove
    this restriction.

3. Minimum driver required is 346.41.

    This version of SASSI is designed to work with the CUDA 7 toolchain,
    which also has that requirement.



