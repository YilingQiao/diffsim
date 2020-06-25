###############################################################################
# Uncomment for debugging
# DEBUG := 1
# Pretty build
# Q ?= @

CXX := g++

# PYTHON Header path
PYTHON_HEADER_DIR := $(shell python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')
PYTORCH_INCLUDES := $(shell python -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')
PYTORCH_LIBRARIES := $(shell python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')

# CUDA ROOT DIR that contains bin/ lib64/ and include/
# CUDA_DIR := /usr/local/cuda

ARCSIM_DIR = arcsim

INCLUDE_DIRS += $(PYTHON_HEADER_DIR)
INCLUDE_DIRS += $(PYTORCH_INCLUDES)
INCLUDE_DIRS += $(ARCSIM_DIR)/dependencies/include
INCLUDE_DIRS += $(ARCSIM_DIR)/src

# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

###############################################################################
SRC_DIR := ./$(ARCSIM_DIR)/src
OBJ_DIR := ./$(ARCSIM_DIR)/objs
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
STATIC_LIB := $(OBJ_DIR)/libmake_pytorch.a

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# We will also explicitly add stdc++ to the link target.
LIBRARIES += stdc++ cudart c10 caffe2 torch torch_python caffe2_gpu

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
	# https://gcoe-dresden.de/reaching-the-shore-with-a-fog-warning-my-eurohack-day-4-morning-session/
	NVCCFLAGS += -g -G # -rdc true
else
	COMMON_FLAGS += -DNDEBUG -O3
endif

WARNINGS := -Wall -Wno-sign-compare -Wcomment

INCLUDE_DIRS += $(BLAS_INCLUDE)

# Automatic dependency generation (nvcc is handled separately)
CXXFLAGS += -MMD -MP

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) \
	     -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=0
CXXFLAGS += -pthread -fPIC -fwrapv -std=c++11 $(COMMON_FLAGS) $(WARNINGS) -fopenmp

all: $(STATIC_LIB)
	rm -rf ./build
	python setup.py install --force

$(OBJ_DIR):
	@ mkdir -p $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@ echo CXX $<
	$(Q)$(CXX) $< $(CXXFLAGS) -c -o $@

$(STATIC_LIB): $(OBJS) | $(OBJ_DIR)
	$(RM) -f $(STATIC_LIB)
	$(RM) -rf build dist
	@ echo LD -o $@
	ar rc $(STATIC_LIB) $(OBJS)
	rm objs
	ln -s $(OBJ_DIR) objs

clean:
	@- $(RM) -rf $(OBJ_DIR) build dist
