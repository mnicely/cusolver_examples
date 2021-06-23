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
#include <cstring>
#include <stdexcept>

#include <curand.h>

#include "magma_v2.h"

#include "utilities.h"

#define VERIFY 0

template<typename T, typename U>
void SingleGPUManaged( const int &ngpu, const int &loops, const U &N, const U &lda, const U &ldb, T *A, T *B ) {

#if VERIFY
    T *B_input {};
    T *A_input {};

    size_t sizeBytesA { sizeof( T ) * lda * N };
    size_t sizeBytesB { sizeof( T ) * N };

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

    std::printf( "Pivot is on : compute P*A = L*U\n" );

    magma_int_t info {}; /* error info */

    U *d_Ipiv { nullptr }; /* pivoting sequence */
    if ( MAGMA_SUCCESS != magma_imalloc_cpu( &d_Ipiv, N ) ) {
        throw std::runtime_error( "Error allocating d_Ipiv\n" );
    }

    // Check GPU memory used on single GPU
    CheckMemoryUsed( 1 );

    std::printf( "\nRunning GETRF\n" );

    CUDA_RT_CALL( cudaEventRecord( startEvent ) );

    for ( int i = 0; i < loops; i++ ) {

        /* step 4: LU factorization */
        if ( ngpu > 1 ) {
            CUDA_RT_CALL( magma_zgetrf_m( ngpu, N, N, A, lda, d_Ipiv, &info ) );
        } else {
            CUDA_RT_CALL( magma_zgetrf( N, N, A, lda, d_Ipiv, &info ) );
        }

        if ( info != 0 ) {
            throw std::runtime_error( std::to_string( -info ) + "-th parameter is wrong (magma_zgetrf) \n" );
        }
    }

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

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

    if ( MAGMA_SUCCESS != magma_free_cpu( d_Ipiv ) ) {
        throw std::runtime_error( "Error freeing d_Ipiv\n" );
    }
    CUDA_RT_CALL( cudaEventDestroy( startEvent ) );
    CUDA_RT_CALL( cudaEventDestroy( stopEvent ) );
#if VERIFY
    CUDA_RT_CALL( cudaFree( A_input ) );
    CUDA_RT_CALL( cudaFree( B_input ) );
#endif
}

int main( int argc, char *argv[] ) {

    if ( MAGMA_SUCCESS != magma_init( ) ) {
        throw std::runtime_error( "Error magma_init\n" );
    }

    magma_int_t m {};
    magma_int_t loops {};
    if ( argc < 3 ) {
        m     = 512;
        loops = 5;
    } else {
        m     = std::atoi( argv[1] );
        loops = std::atoi( argv[2] );
    }

    magma_int_t ngpu = magma_num_gpus( );
    std::printf( "Magma sees %llu GPUs\n", ngpu );

    const magma_int_t lda { m };
    const magma_int_t ldb { m };

    using data_type = magmaDoubleComplex;

    data_type *temp_A {};
    data_type *temp_B {};

    size_t sizeA { static_cast<size_t>( lda ) * m };
    size_t sizeB { static_cast<size_t>( m ) };

    CUDA_RT_CALL( cudaMallocManaged( &temp_A, sizeof( data_type ) * sizeA ) );
    CUDA_RT_CALL( cudaMallocManaged( &temp_B, sizeof( data_type ) * sizeB ) );

    // Generate random numbers on the GPU
    CreateRandomData( "A", sizeA * 2, reinterpret_cast<double *>( temp_A ) );
    CreateRandomData( "B", sizeB * 2, reinterpret_cast<double *>( temp_B ) );

    data_type *m_A {};
    data_type *m_B {};

    if ( MAGMA_SUCCESS != magma_zmalloc_pinned( &m_A, sizeA ) ) {
        throw std::runtime_error( "Error allocating A\n" );
    }
    if ( MAGMA_SUCCESS != magma_zmalloc_pinned( &m_B, sizeB ) ) {
        throw std::runtime_error( "Error allocating B\n" );
    }

    std::memcpy( m_A, temp_A, sizeof( data_type ) * sizeA );
    std::memcpy( m_B, temp_B, sizeof( data_type ) * sizeB );

    // Free memory
    CUDA_RT_CALL( cudaFree( temp_A ) );
    CUDA_RT_CALL( cudaFree( temp_B ) );

    // Managed Memory
    for ( int i = 1; i < ( ngpu * 2 ); i *= 2 ) {
        std::printf( "\n\n******************************************\n" );
        std::printf( "Run Warmup w/ %d GPUs\n", i );
        SingleGPUManaged( ngpu, 1, m, lda, ldb, m_A, m_B );

        std::printf( "\n\n******************************************\n" );
        std::printf( "Run LU Decomposition w/ %d GPUs\n", i );
        SingleGPUManaged( ngpu, loops, m, lda, ldb, m_A, m_B );
    }

    if ( MAGMA_SUCCESS != magma_free_pinned( m_A ) ) {
        throw std::runtime_error( "Error freeing A\n" );
    }

    if ( MAGMA_SUCCESS != magma_free_pinned( m_B ) ) {
        throw std::runtime_error( "Error freeing B\n" );
    }

    return ( EXIT_SUCCESS );
}