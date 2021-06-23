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

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <curand.h>
#include <cusolverDn.h>

#include "utilities.h"

#define VERIFY 0

template<typename T>
void SingleGPUManaged( const int &device,
                       const int &loops,
                       const int &algo,
                       const int &N,
                       const int &lda,
                       const int &ldb,
                       T *        A,
                       T *        B ) {

    size_t sizeBytesA { sizeof( T ) * lda * N };
    size_t sizeBytesB { sizeof( T ) * N };

#if VERIFY
    T *B_input {};
    T *A_input {};

    std::printf( "Allocating space for verification\n" );
    CUDA_RT_CALL( cudaMallocManaged( &A_input, sizeBytesA ) );
    CUDA_RT_CALL( cudaMallocManaged( &B_input, sizeBytesB ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( A_input, sizeBytesA, cudaCpuDeviceId, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( B_input, sizeBytesB, cudaCpuDeviceId, NULL ) );

    std::printf( "Copy A to A_input: Needed for verification\n" );
    CUDA_RT_CALL( cudaMemcpy( A_input, A, sizeBytesA, cudaMemcpyDeviceToHost ) );
    std::printf( "Copy B to B_input: Needed for verification\n" );
    CUDA_RT_CALL( cudaMemcpy( B_input, B, sizeBytesB, cudaMemcpyDeviceToHost ) );
#endif

    CUDA_RT_CALL( cudaMemPrefetchAsync( A, sizeBytesA, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( B, sizeBytesB, device, NULL ) );

    // Start timer
    cudaEvent_t startEvent { nullptr };
    cudaEvent_t stopEvent { nullptr };
    float       elapsed_gpu_ms {};

    CUDA_RT_CALL( cudaEventCreate( &startEvent, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stopEvent, cudaEventBlockingSync ) );

    std::printf( "Pivot is on : compute P*A = L*U\n" );

    /* step 1: create cusolver handle, bind a stream */
    cusolverDnHandle_t cusolverH { nullptr };
    CUDA_RT_CALL( cusolverDnCreate( &cusolverH ) );

    // Create stream
    cudaStream_t stream {};
    CUDA_RT_CALL( cudaStreamCreate( &stream ) );
    CUDA_RT_CALL( cusolverDnSetStream( cusolverH, stream ) );

    /* step 2: copy A to device */
    int *d_info { nullptr }; /* error info */
    CUDA_RT_CALL( cudaMallocManaged( &d_info, sizeof( int ) ) );

    int64_t *d_Ipiv { nullptr }; /* pivoting sequence */
    CUDA_RT_CALL( cudaMallocManaged( &d_Ipiv, sizeof( int64_t ) * N ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( d_Ipiv, sizeof( int64_t ) * N, device, stream ) );

    void *bufferOnDevice { nullptr };
    void *bufferOnHost { nullptr };

    size_t workspaceInBytesOnDevice {};
    size_t workspaceInBytesOnHost {};

    CUDA_RT_CALL( cusolverDnXgetrf_bufferSize(
        cusolverH, NULL, N, N, CUDA_C_64F, A, lda, CUDA_C_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost ) );

    CheckMemoryUsed( 1 );

    std::printf( "\nAllocate device workspace, lwork = %lu\n", workspaceInBytesOnDevice );
    std::printf( "Allocate host workspace, lwork = %lu\n\n", workspaceInBytesOnHost );

    CUDA_RT_CALL( cudaMallocManaged( &bufferOnDevice, workspaceInBytesOnDevice ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( bufferOnDevice, workspaceInBytesOnDevice, device, stream ) );
    CUDA_RT_CALL(
        cudaMemAdvise( bufferOnDevice, workspaceInBytesOnDevice, cudaMemAdviseSetPreferredLocation, device ) );
    CUDA_RT_CALL( cudaMemAdvise( bufferOnDevice, workspaceInBytesOnDevice, cudaMemAdviseSetAccessedBy, device ) );

    if ( 0 < workspaceInBytesOnHost ) {
        CUDA_RT_CALL( cudaMallocManaged( &bufferOnHost, workspaceInBytesOnHost ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( bufferOnHost, workspaceInBytesOnHost, cudaCpuDeviceId, NULL ) );
        assert( NULL != bufferOnHost );
    }

    // Create advanced params
    cusolverDnParams_t params;
    CUDA_RT_CALL( cusolverDnCreateParams( &params ) );
    if ( algo == 0 ) {
        std::printf( "Using New Algo\n" );
        CUDA_RT_CALL( cusolverDnSetAdvOptions( params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0 ) );
    } else {
        std::printf( "Using Legacy Algo\n" );
        CUDA_RT_CALL( cusolverDnSetAdvOptions( params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1 ) );
    }

    // Check GPU memory used on single GPU
    CheckMemoryUsed( 1 );

    CUDA_RT_CALL( cudaMemAdvise( A, sizeBytesA, cudaMemAdviseSetPreferredLocation, device ) );
    CUDA_RT_CALL( cudaMemAdvise( A, sizeBytesA, cudaMemAdviseSetAccessedBy, device ) );

    CUDA_RT_CALL( cudaMemAdvise( B, sizeBytesB, cudaMemAdviseSetPreferredLocation, device ) );
    CUDA_RT_CALL( cudaMemAdvise( B, sizeBytesB, cudaMemAdviseSetAccessedBy, device ) );

    std::printf( "\nRunning GETRF\n" );

    CUDA_RT_CALL( cudaEventRecord( startEvent ) );

    for ( int i = 0; i < loops; i++ ) {

        /* step 4: LU factorization */
        CUDA_RT_CALL( cusolverDnXgetrf( cusolverH,
                                        params,
                                        static_cast<int64_t>( N ),
                                        static_cast<int64_t>( N ),
                                        CUDA_C_64F,
                                        A,
                                        static_cast<int64_t>( lda ),
                                        d_Ipiv,
                                        CUDA_C_64F,
                                        bufferOnDevice,
                                        workspaceInBytesOnDevice,
                                        bufferOnHost,
                                        workspaceInBytesOnHost,
                                        d_info ) );

        // Must be here to retrieve d_info
        CUDA_RT_CALL( cudaStreamSynchronize( stream ) );

        if ( *d_info ) {
            throw std::runtime_error( std::to_string( -*d_info ) + "-th parameter is wrong (cusolverDnDgetrf) \n" );
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
    CUDA_RT_CALL( cudaMemPrefetchAsync( B, sizeBytesB, cudaCpuDeviceId, stream ) );

    // Calculate Residual Error
    CalculateResidualError( N,
                            lda,
                            reinterpret_cast<double *>( A_input ),
                            reinterpret_cast<double *>( B_input ),
                            reinterpret_cast<double *>( B ) );
#endif

    CUDA_RT_CALL( cudaFree( d_Ipiv ) );
    CUDA_RT_CALL( cudaFree( d_info ) );
    CUDA_RT_CALL( cudaFree( bufferOnDevice ) );
    CUDA_RT_CALL( cudaFree( bufferOnHost ) );
    CUDA_RT_CALL( cusolverDnDestroy( cusolverH ) );
    CUDA_RT_CALL( cudaStreamDestroy( stream ) );

    CUDA_RT_CALL( cudaEventDestroy( startEvent ) );
    CUDA_RT_CALL( cudaEventDestroy( stopEvent ) );

#if VERIFY
    CUDA_RT_CALL( cudaFree( A_input ) );
    CUDA_RT_CALL( cudaFree( B_input ) );
#endif
}

int main( int argc, char *argv[] ) {

    int m {};
    int loops {};
    int algo {};

    if ( argc < 4 ) {
        m     = 512;
        loops = 5;
        algo  = 0;
    } else {
        m     = std::atoi( argv[1] );
        loops = std::atoi( argv[2] );
        algo  = std::atoi( argv[3] );
        if ( algo > 1 || algo < 0 )
            algo = 1;
    }

    int device = -1;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    std::printf( "\ncuSolver: SingleGPUManaged GETRF: N = %d\n\n", m );

    const int lda { m };
    const int ldb { m };

    using data_type = cuDoubleComplex;

    data_type *m_A {};
    data_type *m_B {};

    size_t sizeA { static_cast<size_t>( lda ) * m };
    size_t sizeB { static_cast<size_t>( m ) };

    CUDA_RT_CALL( cudaMallocManaged( &m_A, sizeof( data_type ) * sizeA ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_B, sizeof( data_type ) * sizeB ) );

    // Generate random numbers on the GPU
    // Convert to double and double the number of items for cuRand
    CreateRandomData( "A", sizeA * 2, reinterpret_cast<double *>( m_A ) );
    CreateRandomData( "B", sizeB * 2, reinterpret_cast<double *>( m_B ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    // Managed Memory
    std::printf( "\n\n******************************************\n" );
    std::printf( "Run Warmup\n" );
    SingleGPUManaged( device, 1, algo, m, lda, ldb, m_A, m_B );

    std::printf( "\n\n******************************************\n" );
    std::printf( "Run LU Decomposition\n" );
    SingleGPUManaged( device, loops, algo, m, lda, ldb, m_A, m_B );

    return ( EXIT_SUCCESS );
}