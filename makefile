NVCC	:=nvcc --cudart=static -ccbin g++
CFLAGS	:=-O3 -std=c++11

INC_DIR	:=
LIB_DIR	:=
LIBS	:=-lcusolver -lcusolverMg -lcurand

ARCHES :=-gencode arch=compute_70,code=\"compute_70,sm_70\" \
		-gencode arch=compute_75,code=\"compute_75,sm_75\" \
		-gencode arch=compute_80,code=\"compute_80,sm_80\"

SOURCES :=lu_decomposition

all: $(SOURCES)
.PHONY: all

lu_decomposition: lu_decomposition.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)

clean:
	rm -f $(SOURCES)