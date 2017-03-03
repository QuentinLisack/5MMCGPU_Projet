
ifndef CXX
	CXX = g++
endif

CXXFLAGS = -O3 -lrt

LIBS = -lm

CXX_COMMON = ../../CXX_common

INC = -I $(CXX_COMMON)

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CXX = clang++
	CXXFLAGS = -O3 -std=c++11 -stdlib=libc++
endif

pi: pi.cpp
	$(CXX) $^ $(INC) $(CXXFLAGS) $(LIBS) -o $@

clean:
	rm -f pi
