// Redistribution and use in source and binary forms, with or without modification, are permitted
// provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright notice, this list of
//       conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright notice, this list of
//       conditions and the following disclaimer in the documentation and/or other materials
//       provided with the distribution.
//     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
//       to endorse or promote products derived from this software without specific prior written
//       permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>
#include <stdexcept>
#include <string>

#include <curand.h>
#include <cusolverDn.h>
#include <cusolverMg.h>

#include "utilities.h"

#define VERIFY 0

template<typename T>
void MultiGPU( const int &    num_devices,
               int *          device_list,
               const int64_t &N,
               const int64_t &lda,
               const int64_t &ldb,
               T *            A,
               T *            B,
               T *            X ) {

    std::printf( "\ncuSolverMg: MultiGPU GETRF w/ %d GPUs\n", num_devices );

    std::printf( "Initial memory usage\n" );
    CheckMemoryUsed( num_devices );

    // Start timer
    cudaEvent_t startEvent { nullptr };
    cudaEvent_t stopEvent { nullptr };
    float       elapsed_gpu_ms {};

    CUDA_RT_CALL( cudaEventCreate( &startEvent, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stopEvent, cudaEventBlockingSync ) );

    cusolverMgHandle_t cusolverMgH { nullptr };

    const int IA { 1 };
    const int JA { 1 };
    const int T_A { 256 }; /* tile size of A */

    const int IB { 1 };
    const int JB { 1 };
    const int T_B { 100 }; /* tile size of B */

    int info {};

    cudaLibMgMatrixDesc_t   descrA;
    cudaLibMgMatrixDesc_t   descrB;
    cudaLibMgGrid_t         gridA;
    cudaLibMgGrid_t         gridB;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    std::printf( "\nCreate Mg handle and select devices\n" );
    CUDA_RT_CALL( cusolverMgCreate( &cusolverMgH ) );
    CUDA_RT_CALL( cusolverMgDeviceSelect( cusolverMgH, num_devices, device_list ) );

    std::printf( "\nCreate matrix descriptors for A and B\n" );
    CUDA_RT_CALL( cusolverMgCreateDeviceGrid( &gridA, 1, num_devices, device_list, mapping ) );
    CUDA_RT_CALL( cusolverMgCreateDeviceGrid( &gridB, 1, num_devices, device_list, mapping ) );

    /* (global) A is N-by-N */
    CUDA_RT_CALL( cusolverMgCreateMatrixDesc( &descrA,
                                              N,   /* number of rows of (global) A */
                                              N,   /* number of columns of (global) A */
                                              N,   /* number or rows in a tile */
                                              T_A, /* number of columns in a tile */
                                              CUDA_R_64F,
                                              gridA ) );

    /* (global) B is N-by-1 */
    CUDA_RT_CALL( cusolverMgCreateMatrixDesc( &descrB,
                                              N,         /* number of rows of (global) B */
                                              1, /* number of columns of (global) B */
                                              N,         /* number or rows in a tile */
                                              T_B,       /* number of columns in a tile */
                                              CUDA_R_64F,
                                              gridB ) );

    std::printf( "\nAllocate distributed matrices A, B and IPIV\n" );
    std::vector<T *>   array_d_A( num_devices );
    std::vector<T *>   array_d_B( num_devices );
    std::vector<int *> array_d_IPIV( num_devices );

    /* A := 0 */
    CreateMat( num_devices,
               device_list,
               N,   /* number of columns of global A */
               T_A, /* number of columns per column tile */
               lda, /* leading dimension of local A */
               array_d_A.data( ) );
    /* B := 0 */
    CreateMat( num_devices,
               device_list,
               1, /* number of columns of global B */
               T_B,       /* number of columns per column tile */
               ldb,       /* leading dimension of local B */
               array_d_B.data( ) );
    /* IPIV := 0, IPIV is consistent with A */
    CreateMat( num_devices,
               device_list,
               N,   /* number of columns of global IPIV */
               T_A, /* number of columns per column tile */
               1,   /* leading dimension of local IPIV */
               array_d_IPIV.data( ) );

    std::printf( "\nPrepare data on devices\n" );

    /* distribute A to array_d_A */
    MemcpyH2D( num_devices,
               device_list,
               N,
               N, /* input */
               A,
               lda,               /* output */
               N,                 /* number of columns of global A */
               T_A,               /* number of columns per column tile */
               lda,               /* leading dimension of local A */
               array_d_A.data( ), /* host pointer array of dimension num_devices */
               IA,
               JA );

    /* distribute B to array_d_B */
    MemcpyH2D( num_devices,
               device_list,
               N,
               1, /* input */
               B,
               ldb,               /* output */
               1,         /* number of columns of global B */
               T_B,               /* number of columns per column tile */
               ldb,               /* leading dimension of local B */
               array_d_B.data( ), /* host pointer array of dimension num_devices */
               IB,
               JB );

    std::printf( "\nAllocate workspace space\n" );
    int64_t lwork_getrf {};
    int64_t lwork_getrs {};
    int64_t lwork {}; /* workspace: number of elements per device */

    CUDA_RT_CALL( cusolverMgGetrf_bufferSize( cusolverMgH,
                                              N,
                                              N,
                                              reinterpret_cast<void **>( array_d_A.data( ) ),
                                              IA, /* base-1 */
                                              JA, /* base-1 */
                                              descrA,
                                              array_d_IPIV.data( ),
                                              CUDA_R_64F,
                                              &lwork_getrf ) );

    CUDA_RT_CALL( cusolverMgGetrs_bufferSize( cusolverMgH,
                                              CUBLAS_OP_N,
                                              N,
                                              1, /* NRHS */
                                              reinterpret_cast<void **>( array_d_A.data( ) ),
                                              IA,
                                              JA,
                                              descrA,
                                              array_d_IPIV.data( ),
                                              reinterpret_cast<void **>( array_d_B.data( ) ),
                                              IB,
                                              JB,
                                              descrB,
                                              CUDA_R_64F,
                                              &lwork_getrs ) );

    lwork = ( lwork_getrf > lwork_getrs ) ? lwork_getrf : lwork_getrs;

    std::printf( "\tAllocate device workspace, lwork: %lu (bytes)\n\n", lwork * sizeof( int64_t ) );

    std::vector<T *> array_d_work( num_devices );

    /* array_d_work[j] points to device workspace of device j */
    WorkspaceAlloc( num_devices,
                    device_list,
                    sizeof( int64_t ) * lwork, /* number of bytes per device */
                    array_d_work.data( ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) ); /* sync all devices */

    CheckMemoryUsed( num_devices );

    CUDA_RT_CALL( cudaEventRecord( startEvent ) );

    std::printf( "\nSolve A*X = B: GETRF and GETRS\n" );
    CUDA_RT_CALL( cusolverMgGetrf( cusolverMgH,
                                   N,
                                   N,
                                   reinterpret_cast<void **>( array_d_A.data( ) ),
                                   IA,
                                   JA,
                                   descrA,
                                   array_d_IPIV.data( ),
                                   CUDA_R_64F,
                                   reinterpret_cast<void **>( array_d_work.data( ) ),
                                   lwork,
                                   &info /* host */ ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) ); /* sync all devices */

    if ( info ) {
        throw std::runtime_error( std::to_string( -info ) + "-th parameter is wrong (cusolverMgGetrf) \n" );
    }

    CUDA_RT_CALL( cusolverMgGetrs( cusolverMgH,
                                   CUBLAS_OP_N,
                                   N,
                                   1, /* NRHS */
                                   reinterpret_cast<void **>( array_d_A.data( ) ),
                                   IA,
                                   JA,
                                   descrA,
                                   array_d_IPIV.data( ),
                                   reinterpret_cast<void **>( array_d_B.data( ) ),
                                   IB,
                                   JB,
                                   descrB,
                                   CUDA_R_64F,
                                   reinterpret_cast<void **>( array_d_work.data( ) ),
                                   lwork,
                                   &info /* host */ ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) ); /* sync all devices */

    if ( info ) {
        throw std::runtime_error( std::to_string( -info ) + "-th parameter is wrong (cusolverMgGetrs) \n" );
    }

    // Stop timer
    CUDA_RT_CALL( cudaEventRecord( stopEvent ) );
    CUDA_RT_CALL( cudaEventSynchronize( stopEvent ) );

    CUDA_RT_CALL( cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent ) );
    std::printf( "\nRuntime = %0.2f ms\n\n", elapsed_gpu_ms );

#if VERIFY
    std::printf( "Retrieve solution vector X\n" );
    MemcpyD2H( num_devices,
               device_list,
               N,
               1,         /* input */
               1, /* number of columns of global B */
               T_B,       /* number of columns per column tile */
               ldb,       /* leading dimension of local B */
               array_d_B.data( ),
               IB,
               JB, /* output */
               X,  /* N-by-1 */
               ldb );

    // Calculate Residual Error
    CalculateResidualError( N, lda, A, B, X );
#endif

    std::printf( "Free resources\n" );
    CUDA_RT_CALL( cudaEventDestroy( startEvent ) );
    CUDA_RT_CALL( cudaEventDestroy( stopEvent ) );
    // CUDA_RT_CALL( cusolverMgDestroy( cusolverMgH ) );
    CUDA_RT_CALL( cusolverMgDestroyMatrixDesc( descrA ) );
    CUDA_RT_CALL( cusolverMgDestroyMatrixDesc( descrB ) );
    CUDA_RT_CALL( cusolverMgDestroyGrid( gridA ) );
    CUDA_RT_CALL( cusolverMgDestroyGrid( gridB ) );

    DestroyMat( num_devices,
                device_list,
                N,   /* number of columns of global A */
                T_A, /* number of columns per column tile */
                array_d_A.data( ) );
    DestroyMat( num_devices,
                device_list,
                1, /* number of columns of global B */
                T_B,       /* number of columns per column tile */
                array_d_B.data( ) );
    DestroyMat( num_devices,
                device_list,
                N,   /* number of columns of global IPIV */
                T_A, /* number of columns per column tile */
                array_d_IPIV.data( ) );

    WorkspaceFree( num_devices, device_list, array_d_work.data( ) );
}

int main( int argc, char *argv[] ) {

    int64_t m = 512;
    if ( argc > 1 )
        m = std::atoi( argv[1] );

    // Setup for MultiGPU version
    int num_devices {};
    CUDA_RT_CALL( cudaGetDeviceCount( &num_devices ) );
    std::vector<int> device_list( num_devices );

    GetDeviceProperties( num_devices, device_list.data( ) );

    std::printf( "Enable peer access\n" );
    EnablePeerAccess( num_devices );

    const int64_t lda { m };
    const int64_t ldb { m };

    using data_type = double;

    data_type *m_A {};
    data_type *m_B {};
    data_type *m_X {};

    CUDA_RT_CALL( cudaMallocManaged( &m_A, sizeof( data_type ) * lda * m ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_B, sizeof( data_type ) * m ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_X, sizeof( data_type ) * m ) );

    // // Generate random numbers on the CPU
    CreateRandomData( cudaCpuDeviceId, "A", m * lda, m_A );
    CreateRandomData( cudaCpuDeviceId, "B", m, m_B );

    // Managed Memory
    std::printf( "\nRun LU Decomposition\n" );
    for ( int i = 1; i < ( num_devices * 2 ); i *= 2 ) {
        
        MultiGPU( i, device_list.data( ), m, lda, ldb, m_A, m_B, m_X );
    }

    CUDA_RT_CALL( cudaFree( m_A ) );
    CUDA_RT_CALL( cudaFree( m_B ) );
    CUDA_RT_CALL( cudaFree( m_X ) );

    return ( EXIT_SUCCESS );
}