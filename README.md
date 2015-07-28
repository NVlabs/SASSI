This project contains the SASSI instrumentation tool.  SASSI is not
part of the official CUDA toolkit, but instead is a research prototype
from the Architecture Research Group at NVIDIA.

Directory Structure
--------------------

This directory contains the SASSI release, which is comprised of the
following items:

tool:     This directory contains the SASSI-enabled ptxas binary for a
	  number of targets, the SASSI header files, and an
	  installation script.


instlibs: Sample instrumentatation libraries.  Note: the samples rely
	  on a modified version of halloc, which we have included.


doc: 	  Documentation including the SASSI user's guide and ISCA paper.


Installation
------------------

To install SASSI, please follow the instructions in the user guide,
which you can find at ./doc/sassi-user-guide.pdf.  If you are really
impatient, find and execute the binary installer for your platform in
the ./tool directory.

Restrictions
------------------

1. 32-bit architectures are not supported.

This was an early design decision to reduce the large cross product of
possible configurations.  Please let us know if 32-bit support would
be useful though, because it probably wouldn't be too hard to
support.

2. Programs currently have to be compiled with "-rdc=true", which
affects performance.

SASSI allows users to instrument code by injecting function calls to
user-defined functions that are later linked in.  In order to perform
cross-module function calls in CUDA one must use the "relocatable
device code" option, "-rdc=true".  Future versions of SASSI may remove
this restriction.

3. Minimum driver required is 346.41.

This version of SASSI is designed to work with the CUDA 7 toolchain,
which also has that requirement.

4. Compiler-inserted MEMBARs and TEXDEPBARs are not accounted for.

