
ifndef CXX
	CXX=g++
endif

CXX_COMMON = ../../CXX_common

CXXFLAGS=

INC = -I $(CXX_COMMON)

LIBS = -lOpenCL

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CXX = clang++
	LIBS = -framework OpenCL
endif

DeviceInfo: DeviceInfo.cpp
	$(CXX) $^ $(INC) $(CXXFLAGS) $(LIBS) -o $@


clean:
	rm -f DeviceInfo
