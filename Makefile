# =============================================================================
#  FILE        : Makefile
#  PROJECT     : TPGS Single-Dish 40.0 Gsps Spectrometer Pipeline
#  DESCRIPTION : Build automation for NVIDIA GH200 (SM90) and Blackwell (SM120)
#  AUTHOR      : Jongsoo Kim, Korea Astronomy and Space Science Institute (KASI)
#  VERSION     : 0.1.0
#  DATE        : October 2026
#  OBSERVATORY : ALMA Wideband Sensitivity Upgrade (WSU)
# =============================================================================

NVCC      := nvcc
CXXFLAGS  := -O3 -std=c++17 --ptxas-options=-v -lineinfo
MATHDX    ?= /opt/nvidia/mathdx/26.03/include
INCLUDES  := -I ./include -I $(MATHDX)

# Target Architecture: Default to Blackwell RTX PRO 6000 (sm_120)
ARCH_CODE ?= 1200
ifeq ($(ARCH_CODE),1200)
    GENCODE := -gencode arch=compute_120,code=sm_120 -DTARGET_ARCH=1200
else ifeq ($(ARCH_CODE),900)
    GENCODE := -gencode arch=compute_90,code=sm_90 -DTARGET_ARCH=900
endif

SRC       := src/tpgs_pipeline.cu
BIN       := bin/tpgs_pipeline

.PHONY: all run clean distclean sm120 sm90

all: $(BIN)

sm120:
	$(MAKE) ARCH_CODE=1200

sm90:
	$(MAKE) ARCH_CODE=900

$(BIN): $(SRC) include/tpgs_params.h include/packet_wsu.h
	@mkdir -p bin data plots
	$(NVCC) $(CXXFLAGS) $(GENCODE) $(INCLUDES) -o $@ $<

run: $(BIN)
	./$(BIN)

clean:
	rm -rf bin/ *.log

distclean: clean
	rm -rf data/*.bin
