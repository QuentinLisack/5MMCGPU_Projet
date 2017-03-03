
ifndef CXX
	CXX=g++
endif

CXXFLAGS=-O3 -ffast-math

OPENCLPATH = /opt/AMDAPPSDK-3.0

LIBS = -I$(OPENCLPATH)/include -L$(OPENCLPATH)/lib/x86_64 -lOpenCL -lrt

COMMON_DIR = ../../CXX_common

INC = -I $(COMMON_DIR)

MMUL_OBJS = matmul.o matrix_lib.o wtime.o
EXEC = mult

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CXX = clang++
	CXXFLAGS += -stdlib=libc++
	LIBS = -lm -framework OpenCL
endif

all: $(EXEC)

mult: $(MMUL_OBJS)
	$(CXX) $(MMUL_OBJS) $(CXXFLAGS) $(LIBS) -o $(EXEC)

wtime.o: $(COMMON_DIR)/wtime.c
	$(CXX) -c $^ $(CXXFLAGS) -o $@


.c.o:
	$(CXX) -c $< $(CXXFLAGS) -o $@

.cpp.o:
	$(CXX) -c $< $(CXXFLAGS) $(INC) -o $@

matmul.o:	matmul.hpp matrix_lib.hpp

matrix_lib.o:	matmul.hpp

clean:
	rm -f $(MMUL_OBJS) $(EXEC)
