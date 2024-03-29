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
void MultiGPU( const int &num_devices,
               int *      device_list,
               const int &loops,
               const int &N,
               const int &lda,
               const int &ldb,
               T *        A,
               T *        B,
               T *        X ) {

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
                                              CUDA_C_64F,
                                              gridA ) );

    /* (global) B is N-by-1 */
    CUDA_RT_CALL( cusolverMgCreateMatrixDesc( &descrB,
                                              N,   /* number of rows of (global) B */
                                              1,   /* number of columns of (global) B */
                                              N,   /* number or rows in a tile */
                                              T_B, /* number of columns in a tile */
                                              CUDA_C_64F,
                                              gridB ) );

    std::printf( "\nAllocate distributed matrices A, B and IPIV\n" );
    std::vector<T *>   array_d_A( num_devices );
    std::vector<T *>   array_d_B( num_devices );
    std::vector<int *> array_d_IPIV( num_devices );

    /* A := 0 */
    std::printf( "Create A\n" );
    CreateMat( num_devices,
               device_list,
               N,   /* number of columns of global A */
               T_A, /* number of columns per column tile */
               lda, /* leading dimension of local A */
               array_d_A.data( ) );
    /* B := 0 */
    std::printf( "Create B\n" );
    CreateMat( num_devices,
               device_list,
               1,   /* number of columns of global B */
               T_B, /* number of columns per column tile */
               ldb, /* leading dimension of local B */
               array_d_B.data( ) );
    /* IPIV := 0, IPIV is consistent with A */
    std::printf( "Create IPIV\n" );
    CreateMat( num_devices,
               device_list,
               N,   /* number of columns of global IPIV */
               T_A, /* number of columns per column tile */
               1,   /* leading dimension of local IPIV */
               array_d_IPIV.data( ) );

    std::printf( "\nPrepare data on devices\n" );

    /* distribute A to array_d_A */
    std::printf( "Copy A\n" );
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
    std::printf( "Copy B\n" );
    MemcpyH2D( num_devices,
               device_list,
               N,
               1, /* input */
               B,
               ldb,               /* output */
               1,                 /* number of columns of global B */
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
                                              CUDA_C_64F,
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
                                              CUDA_C_64F,
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

    std::printf( "\nRunning GETRF\n" );

    CUDA_RT_CALL( cudaEventRecord( startEvent ) );

    for ( int i = 0; i < loops; i++ ) {

        CUDA_RT_CALL( cusolverMgGetrf( cusolverMgH,
                                       N,
                                       N,
                                       reinterpret_cast<void **>( array_d_A.data( ) ),
                                       IA,
                                       JA,
                                       descrA,
                                       array_d_IPIV.data( ),
                                       CUDA_C_64F,
                                       reinterpret_cast<void **>( array_d_work.data( ) ),
                                       lwork,
                                       &info /* host */ ) );

        CUDA_RT_CALL( cudaDeviceSynchronize( ) ); /* sync all devices */

        if ( info ) {
            throw std::runtime_error( std::to_string( -info ) + "-th parameter is wrong (cusolverMgGetrf) \n" );
        }
    }

    // Stop timer
    CUDA_RT_CALL( cudaEventRecord( stopEvent ) );
    CUDA_RT_CALL( cudaEventSynchronize( stopEvent ) );

    CUDA_RT_CALL( cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent ) );
    double avg { elapsed_gpu_ms / loops };
    double flops { FLOPS_ZGETRF( N, N ) };
    double perf { 1e-9 * flops / avg };
    std::printf( "\nRuntime = %0.2f ms (avg over %d runs) : @ %0.2f GFLOPs\n\n", avg, loops, perf );

#if VERIFY
    std::printf( "Retrieve solution vector X\n" );
    MemcpyD2H( num_devices,
               device_list,
               N,
               1,   /* input */
               1,   /* number of columns of global B */
               T_B, /* number of columns per column tile */
               ldb, /* leading dimension of local B */
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
                1,   /* number of columns of global B */
                T_B, /* number of columns per column tile */
                array_d_B.data( ) );
    DestroyMat( num_devices,
                device_list,
                N,   /* number of columns of global IPIV */
                T_A, /* number of columns per column tile */
                array_d_IPIV.data( ) );

    WorkspaceFree( num_devices, device_list, array_d_work.data( ) );
}

int main( int argc, char *argv[] ) {

    int m {};
    int loops {};
    int ngpu {};

    if ( argc < 4 ) {
        m     = 512;
        loops = 5;
        ngpu  = 1;
    } else {
        m     = std::atoi( argv[1] );
        loops = std::atoi( argv[2] );
        ngpu  = std::atoi( argv[3] );
    }

    // Setup for MultiGPU version
    std::vector<int> device_list( ngpu );

    std::printf( "\ncuSOLVERMg: MultiGPU GETRF: N = %d\n\n", m );

    GetDeviceProperties( ngpu, device_list.data( ) );

    std::printf( "\ncuSolverMg: MultiGPU GETRF w/ %d GPUs: N = %d\n\n", ngpu, m );

    std::printf( "Enable peer access\n" );
    EnablePeerAccess( ngpu );

    const int lda { m };
    const int ldb { m };

    using data_type = cuDoubleComplex;

    data_type *m_A {};
    data_type *m_B {};
    data_type *m_X {};

    size_t sizeA { static_cast<size_t>( lda ) * m };
    size_t sizeB { static_cast<size_t>( m ) };
    size_t sizeX { static_cast<size_t>( m ) };

    CUDA_RT_CALL( cudaMallocManaged( &m_A, sizeof( data_type ) * sizeA ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_B, sizeof( data_type ) * sizeB ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_X, sizeof( data_type ) * sizeX ) );

    // Generate random numbers on the GPU
    // Convert to double and double the number of items for cuRand
    CreateRandomData( "A", sizeA * 2, reinterpret_cast<double *>( m_A ) );
    CreateRandomData( "B", sizeB * 2, reinterpret_cast<double *>( m_B ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    // Managed Memory
    std::printf( "\n\n******************************************\n" );
    std::printf( "Run Warmup w/ %d GPUs\n", ngpu );
    MultiGPU( ngpu, device_list.data( ), 1, m, lda, ldb, m_A, m_B, m_X );

    std::printf( "\n\n******************************************\n" );
    std::printf( "Run LU Decomposition w/ %d GPUs\n", ngpu );
    MultiGPU( ngpu, device_list.data( ), loops, m, lda, ldb, m_A, m_B, m_X );

    CUDA_RT_CALL( cudaFree( m_A ) );
    CUDA_RT_CALL( cudaFree( m_B ) );
    CUDA_RT_CALL( cudaFree( m_X ) );

    return ( EXIT_SUCCESS );
}