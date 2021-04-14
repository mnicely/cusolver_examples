NVCC	:=nvcc --cudart=static -ccbin g++ -Xcompiler -fopenmp
CFLAGS	:=-O3 -std=c++14

INC_DIR	:=-I$(HOME)/workStuff/thrust
LIB_DIR	:=
LIBS	:=-lcusolver -lcusolverMg -lcurand

ARCHES :=-gencode arch=compute_70,code=\"compute_70,sm_70\" \
		-gencode arch=compute_75,code=\"compute_75,sm_75\" \
		-gencode arch=compute_80,code=\"compute_80,sm_80\"

MAGMADIR     := $(HOME)/workStuff/magma
MAGMALIB	 := -L$(MAGMADIR)/lib
MAGMAINC	 := -I$(MAGMADIR)/include
MAGMA_LIBS   := -L$(MAGMADIR)/lib -lmagma

SOURCES :=lu_decomposition_cusolver \
			lu_decomposition_cusolvermg \
			lu_decomposition_magma 

all: $(SOURCES)
.PHONY: all

lu_decomposition_cusolver: lu_decomposition_cusolver.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)

lu_decomposition_cusolvermg: lu_decomposition_cusolvermg.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)

lu_decomposition_magma: lu_decomposition_magma.cu
	$(NVCC) $(CFLAGS) $(MAGMAINC) $(MAGMALIB) ${ARCHES} $^ -o $@ $(MAGMA_LIBS) $(LIBS)

clean:
	rm -f $(SOURCES)