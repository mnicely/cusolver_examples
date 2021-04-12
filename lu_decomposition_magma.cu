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

#include "magma_v2.h"

#include "utilities.h"

#define VERIFY 0

constexpr int pivot_on { 1 };

template<typename T, typename U>
void SingleGPUManaged( const int &device, const U &N, const U &lda, const U &ldb, T *A, T *B ) {

    std::printf( "\ncuSolver: SingleGPUManaged GETRF\n" );

#if VERIFY
    size_t sizeBytesA { sizeof( T ) * lda * N };
    size_t sizeBytesB { sizeof( T ) * N };

    T *B_input {};
    T *A_input {};

    CUDA_RT_CALL( cudaMallocManaged( &A_input, sizeBytesA ) );
    CUDA_RT_CALL( cudaMallocManaged( &B_input, sizeBytesB ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( A_input, sizeBytesA, cudaCpuDeviceId, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( B_input, sizeBytesB, cudaCpuDeviceId, NULL ) );

    CUDA_RT_CALL( cudaMemcpy( A_input, A, sizeBytesA, cudaMemcpyDeviceToHost ) );
    CUDA_RT_CALL( cudaMemcpy( B_input, B, sizeBytesB, cudaMemcpyDeviceToHost ) );
#endif

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

    // Create stream
    cudaStream_t stream {};
    CUDA_RT_CALL( cudaStreamCreate( &stream ) );

    magma_init( );
    magma_queue_t queue;
    magma_queue_create_from_cuda( device, stream, NULL, NULL, &queue );

    /* step 2: copy A to device */
    int *d_info { nullptr }; /* error info */
    CUDA_RT_CALL( cudaMallocManaged( &d_info, sizeof( int ) ) );

    U *d_Ipiv { nullptr }; /* pivoting sequence */
    if ( pivot_on ) {
        CUDA_RT_CALL( cudaMallocManaged( &d_Ipiv, sizeof( U ) * N ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( d_Ipiv, sizeof( U ) * N, device, stream ) );
    }

    // Check GPU memory used on single GPU
    CheckMemoryUsed( 1 );

    CUDA_RT_CALL( cudaEventRecord( startEvent ) );

    cudaDeviceSynchronize( );

    std::printf( "\nSolve A*X = B: GETRF and GETRS\n" );

    /* step 4: LU factorization */
    if ( pivot_on ) {
        CUDA_RT_CALL( magma_dgetrf_gpu( N, N, A, lda, d_Ipiv, d_info ) );
    } else {
        CUDA_RT_CALL( magma_dgetrf_nopiv_gpu( N, N, A, lda, d_info ) );
    }

    // Must be here to retrieve d_info
    magma_queue_sync( queue );

    if ( *d_info ) {
        throw std::runtime_error( std::to_string( -*d_info ) + "-th parameter is wrong (cusolverDnDgetrf) \n" );
    }

    /*
     * step 5: solve A*X = B
     */

    if ( pivot_on ) {
        CUDA_RT_CALL( magma_dgetrs_gpu( MagmaNoTrans,
                                        N,
                                        1, /* nrhs */
                                        A,
                                        lda,
                                        d_Ipiv,
                                        B,
                                        ldb,
                                        d_info ) );
    } else {
        CUDA_RT_CALL( magma_dgetrs_nopiv_gpu( MagmaNoTrans,
                                              N,
                                              1, /* nrhs */
                                              A,
                                              lda,
                                              B,
                                              ldb,
                                              d_info ) );
    }

    // Must be here to retrieve d_info
    magma_queue_sync( queue );

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
    CUDA_RT_CALL( cudaStreamDestroy( stream ) );
    magma_queue_destroy( queue );

    CUDA_RT_CALL( cudaEventDestroy( startEvent ) );
    CUDA_RT_CALL( cudaEventDestroy( stopEvent ) );
#if VERIFY
    CUDA_RT_CALL( cudaFree( A_input ) );
    CUDA_RT_CALL( cudaFree( B_input ) );
#endif
}

int main( int argc, char *argv[] ) {

    magma_int_t m = 512;
    if ( argc > 1 )
        m = std::atoi( argv[1] );

    int device = -1;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    const magma_int_t lda { m };
    const magma_int_t ldb { m };

    double *m_A {};
    double *m_B {};

    CUDA_RT_CALL( cudaMallocManaged( &m_A, sizeof( double ) * lda * m ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_B, sizeof( double ) * m ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( m_A, sizeof( double ) * lda * m, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( m_B, sizeof( double ) * m, device, NULL ) );

    // Generate random numbers on the GPU
    CreateRandomData( device, "A", m * lda, m_A );
    CreateRandomData( device, "B", m, m_B );

    // Managed Memory
    std::printf( "Run LU Decomposition\n" );
    SingleGPUManaged( device, m, lda, ldb, m_A, m_B );

    CUDA_RT_CALL( cudaFree( m_A ) );
    CUDA_RT_CALL( cudaFree( m_B ) );

    return ( EXIT_SUCCESS );
}