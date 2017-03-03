# This makefile will produce all the C binaries
# in their respective directories

C_EXEC = Exercise01/C/DeviceInfo Exercise02/C/vadd \
		Exercise04/C/vadd

CXX_EXEC = Exercise01/CXX/DeviceInfo Exercise03/CXX/vadd \
		Exercise04/CXX/vadd \
		Exercise05/CXX/mult Exercise06/CXX/pi \
		Projet

# Change this variable to specify the device type in all
# the Makefile to the OpenCL device type of choice
DEVICE = CL_DEVICE_TYPE_DEFAULT
export DEVICE

# In case you need to rename the C++ compiler, you can
# do it in bulk here
CXX = g++

ifndef CC
	CC = gcc
endif
export CC

.PHONY : $(C_EXEC) $(CXX_EXEC)

all: $(C_EXEC) $(CXX_EXEC)

$(C_EXEC):
	$(MAKE) -C `dirname $@`

$(CXX_EXEC):
	$(MAKE) -C `dirname $@`

.PHONY : clean
clean:
	for e in $(C_EXEC) $(CXX_EXEC); do $(MAKE) -C `dirname $$e` clean; done
