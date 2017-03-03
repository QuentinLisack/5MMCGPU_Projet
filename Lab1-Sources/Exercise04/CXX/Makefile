
ifndef CXX
	CXX=g++
endif

CXX_COMMON = ../../CXX_common

CXXFLAGS=

INC = -I $(CXX_COMMON)

OPENCLPATH = /usr/lib/x86_64-linux-gnu/beignet

LIBS = -I$(OPENCLPATH)/include -L$(OPENCLPATH)/lib/x86_64 -lOpenCL -lrt

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CXX = clang++
	CXXFLAGS += -stdlib=libc++
	LIBS = -framework OpenCL
endif

CXXFLAGS += -D DEVICE=$(DEVICE)

vadd: vadd.cpp
	$(CXX) $^ $(INC) $(CXXFLAGS) $(LIBS) -o $@


clean:
	rm -f vadd
