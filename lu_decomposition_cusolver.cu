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

#include <curand.h>
#include <cusolverDn.h>

#include "utilities.h"

#define VERIFY 1

constexpr int pivot_on { 1 };

template<typename T>
void SingleGPUManaged( const int &device, const int &N, const int &lda, const int &ldb, T *A, T *B ) {

    size_t sizeBytesA { sizeof( T ) * lda * N };
    size_t sizeBytesB { sizeof( T ) * N };

#if VERIFY
    T *B_input {};
    T *A_input {};

    std::printf("Allocating space for verification\n");
    CUDA_RT_CALL( cudaMallocManaged( &A_input, sizeBytesA ) );
    CUDA_RT_CALL( cudaMallocManaged( &B_input, sizeBytesB ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( A_input, sizeBytesA, cudaCpuDeviceId, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( B_input, sizeBytesB, cudaCpuDeviceId, NULL ) );

    std::printf("Copy A to A_input: Needed for verification\n");
    CUDA_RT_CALL( cudaMemcpy( A_input, A, sizeBytesA, cudaMemcpyDeviceToHost ) );
    std::printf("Copy B to B_input: Needed for verification\n");
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

    if ( pivot_on ) {
        std::printf( "Pivot is on : compute P*A = L*U\n" );
    } else {
        std::printf( "Pivot is off: compute A = L*U (not numerically stable)\n" );
    }

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
    if ( pivot_on ) {
        CUDA_RT_CALL( cudaMallocManaged( &d_Ipiv, sizeof( int64_t ) * N ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( d_Ipiv, sizeof( int64_t ) * N, device, stream ) );
    }

    void *bufferOnDevice { nullptr };
    void *bufferOnHost { nullptr };

    size_t workspaceInBytesOnDevice {};
    size_t workspaceInBytesOnHost {};

    // CUDA_RT_CALL( cusolverDnDgetrf_bufferSize( cusolverH, N, N, A, lda, &lwork ) );
    CUDA_RT_CALL( cusolverDnXgetrf_bufferSize(
        cusolverH, NULL, N, N, CUDA_R_64F, A, lda, CUDA_R_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost ) );

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

    // Check GPU memory used on single GPU
    CheckMemoryUsed( 1 );

    CUDA_RT_CALL( cudaMemAdvise( A, sizeBytesA, cudaMemAdviseSetPreferredLocation, device ) );
    CUDA_RT_CALL( cudaMemAdvise( A, sizeBytesA, cudaMemAdviseSetAccessedBy, device ) );

    CUDA_RT_CALL( cudaMemAdvise( B, sizeBytesB, cudaMemAdviseSetPreferredLocation, device ) );
    CUDA_RT_CALL( cudaMemAdvise( B, sizeBytesB, cudaMemAdviseSetAccessedBy, device ) );

    CUDA_RT_CALL( cudaEventRecord( startEvent ) );

    std::printf( "\nSolve A*X = B: GETRF and GETRS\n" );

    /* step 4: LU factorization */
    if ( pivot_on ) {
        CUDA_RT_CALL( cusolverDnXgetrf( cusolverH,
                                        NULL,
                                        static_cast<int64_t>(N),
                                            static_cast<int64_t>(N),
                                        CUDA_R_64F,
                                        A,
                                        static_cast<int64_t>(lda),
                                        d_Ipiv,
                                        CUDA_R_64F,
                                        bufferOnDevice,
                                        workspaceInBytesOnDevice,
                                        bufferOnHost,
                                        workspaceInBytesOnHost,
                                        d_info ) );
    } else {
        CUDA_RT_CALL( cusolverDnXgetrf( cusolverH,
                                        NULL,
                                        static_cast<int64_t>(N),
                                            static_cast<int64_t>(N),
                                        CUDA_R_64F,
                                        A,
                                        static_cast<int64_t>(lda),
                                        nullptr,
                                        CUDA_R_64F,
                                        bufferOnDevice,
                                        workspaceInBytesOnDevice,
                                        bufferOnHost,
                                        workspaceInBytesOnHost,
                                        d_info ) );
    }

    // Must be here to retrieve d_info
    CUDA_RT_CALL( cudaStreamSynchronize( stream ) );

    if ( *d_info ) {
        throw std::runtime_error( std::to_string( -*d_info ) + "-th parameter is wrong (cusolverDnDgetrf) \n" );
    }

    CUDA_RT_CALL( cudaMemAdvise( A, sizeBytesA, cudaMemAdviseSetReadMostly, device ) );
    CUDA_RT_CALL( cudaMemAdvise( d_Ipiv, sizeof( int64_t ) * N, cudaMemAdviseSetReadMostly, device ) );

    /*
     * step 5: solve A*X = B
     */

    if ( pivot_on ) {
        CUDA_RT_CALL( cusolverDnXgetrs( cusolverH,
                                        NULL,
                                        CUBLAS_OP_N,
                                        static_cast<int64_t>(N),
                                        1, /* nrhs */
                                        CUDA_R_64F,
                                        A,
                                        static_cast<int64_t>(lda),
                                        d_Ipiv,
                                        CUDA_R_64F,
                                        B,
                                        static_cast<int64_t>(ldb),
                                        d_info ) );
    } else {
        CUDA_RT_CALL( cusolverDnXgetrs( cusolverH,
                                        NULL,
                                        CUBLAS_OP_N,
                                        static_cast<int64_t>(N),
                                        1, /* nrhs */
                                        CUDA_R_64F,
                                        A,
                                        static_cast<int64_t>(lda),
                                        nullptr,
                                        CUDA_R_64F,
                                        B,
                                        static_cast<int64_t>(ldb),
                                        d_info ) );
    }

    // Must be here to retrieve d_info
    CUDA_RT_CALL( cudaStreamSynchronize( stream ) );

    if ( *d_info ) {
        throw std::runtime_error( std::to_string( -*d_info ) + "-th parameter is wrong (cusolverDnDgetrs) \n" );
    }

    // Stop timer
    CUDA_RT_CALL( cudaEventRecord( stopEvent ) );
    CUDA_RT_CALL( cudaEventSynchronize( stopEvent ) );

    CUDA_RT_CALL( cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent ) );
    std::printf( "\nRuntime = %0.2f ms\n\n", elapsed_gpu_ms );

#if VERIFY
    CUDA_RT_CALL( cudaMemPrefetchAsync( B, sizeBytesB, cudaCpuDeviceId, stream ) );

    // Calculate Residual Error
    CalculateResidualError( N, lda, A_input, B_input, B );
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

    int m = 512;
    if ( argc > 1 )
        m = std::atoi( argv[1] );

    int device = -1;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    std::printf( "\ncuSolver: SingleGPUManaged GETRF: N = %d\n\n", m );

    const int lda { m };
    const int ldb { m };

    using data_type = double;

    data_type *m_A {};
    data_type *m_B {};

    CUDA_RT_CALL( cudaMallocManaged( &m_A, sizeof( data_type ) * lda * m ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_B, sizeof( data_type ) * m ) );

    // Generate random numbers on the GPU
    CreateRandomData( "A", static_cast<int64_t>(lda) * m, m_A );
    CreateRandomData( "B", m, m_B );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    // // Managed Memory
    std::printf( "Run LU Decomposition\n" );
    SingleGPUManaged( device, m, lda, ldb, m_A, m_B );



    return ( EXIT_SUCCESS );
}