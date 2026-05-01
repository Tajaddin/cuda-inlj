# =============================================================================
# Makefile for GPU Index Nested-Loop Join Implementation
# COP6527 - Computing Massive Parallel Systems
# Fall 2025
# =============================================================================

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust based on your GPU)
# Common architectures:
#   - sm_50: Maxwell (GTX 9xx)
#   - sm_60: Pascal (GTX 10xx)
#   - sm_70: Volta (V100)
#   - sm_75: Turing (RTX 20xx)
#   - sm_80: Ampere (RTX 30xx, A100)
#   - sm_86: Ampere (RTX 30xx mobile)
#   - sm_89: Ada Lovelace (RTX 40xx)
#   - sm_90: Hopper (H100)
CUDA_ARCH = -gencode arch=compute_60,code=sm_60 \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86

# Compiler flags
NVCC_FLAGS = -O3 -std=c++11 $(CUDA_ARCH) -Xcompiler -Wall
DEBUG_FLAGS = -g -G -DDEBUG

# Include paths
INCLUDES = -I./include

# Source and object directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/btree_index.cu \
          $(SRC_DIR)/utils.cu \
          $(SRC_DIR)/main.cu

# Object files
OBJECTS = $(OBJ_DIR)/btree_index.o \
          $(OBJ_DIR)/utils.o \
          $(OBJ_DIR)/main.o

# Target executable
TARGET = $(BIN_DIR)/inlj_test

# =============================================================================
# Build Rules
# =============================================================================

.PHONY: all clean debug run help profile

# Default target
all: directories $(TARGET)

# Create directories
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Link target
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
	@echo "Build complete: $(TARGET)"

# Compile source files
$(OBJ_DIR)/btree_index.o: $(SRC_DIR)/btree_index.cu include/inlj.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/utils.o: $(SRC_DIR)/utils.cu include/inlj.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cu include/inlj.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Debug build
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: clean all
	@echo "Debug build complete"

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Cleaned build artifacts"

# Run the program
run: all
	./$(TARGET)

# Run specific test
run1: all
	./$(TARGET) 1

run2: all
	./$(TARGET) 2

run3: all
	./$(TARGET) 3

run4: all
	./$(TARGET) 4

# Profile with nvprof (if available)
profile: all
	nvprof ./$(TARGET) 2

# Profile with Nsight Compute (newer profiling tool)
ncu-profile: all
	ncu --set full -o profile_report ./$(TARGET) 2

# Help target
help:
	@echo "==================================================================="
	@echo "GPU Index Nested-Loop Join - Build System"
	@echo "==================================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Build the project (default)"
	@echo "  debug      - Build with debug symbols"
	@echo "  clean      - Remove build artifacts"
	@echo "  run        - Build and run all tests"
	@echo "  run1       - Run Test 1: Correctness verification"
	@echo "  run2       - Run Test 2: Performance comparison"
	@echo "  run3       - Run Test 3: Scalability analysis"
	@echo "  run4       - Run Test 4: Selectivity analysis"
	@echo "  profile    - Profile with nvprof"
	@echo "  ncu-profile- Profile with Nsight Compute"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  Edit CUDA_ARCH in Makefile to match your GPU architecture"
	@echo ""
	@echo "==================================================================="

# =============================================================================
# Additional Rules for Development
# =============================================================================

# Check CUDA installation
check-cuda:
	@echo "NVCC version:"
	@$(NVCC) --version
	@echo ""
	@echo "CUDA devices:"
	@nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

# Format check (requires clang-format)
format-check:
	@find $(SRC_DIR) include -name "*.cu" -o -name "*.h" | \
		xargs clang-format --dry-run -Werror 2>/dev/null || \
		echo "clang-format not available or formatting issues found"

# Create source distribution
dist: clean
	tar -czvf cuda_inlj_project.tar.gz \
		Makefile README.md \
		include/ src/ \
		--exclude='*.o' --exclude='*.exe'
	@echo "Distribution archive created: cuda_inlj_project.tar.gz"
