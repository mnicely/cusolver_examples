#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <curand.h>
#include <cusolverDn.h>

#include "utilities.h"

#define VERIFY 0

constexpr int pivot_on { 1 };

template<typename T, typename U>
void SingleGPUManaged( const int &device, const U &N, const U &lda, const U &ldb, T *A, T *B ) {

    std::printf( "\ncuSolver: SingleGPUManaged GETRF\n" );

    size_t sizeBytesA { sizeof( T ) * lda * N };
    size_t sizeBytesB { sizeof( T ) * N };

#if VERIFY
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
    cusolverDnHandle_t cusolverH { nullptr };
    CUDA_RT_CALL( cusolverDnCreate( &cusolverH ) );

    // Create stream
    cudaStream_t stream {};
    CUDA_RT_CALL( cudaStreamCreate( &stream ) );
    CUDA_RT_CALL( cusolverDnSetStream( cusolverH, stream ) );

    /* step 2: copy A to device */
    int *d_info { nullptr }; /* error info */
    CUDA_RT_CALL( cudaMallocManaged( &d_info, sizeof( int ) ) );

    U *d_Ipiv { nullptr }; /* pivoting sequence */
    if ( pivot_on ) {
        CUDA_RT_CALL( cudaMallocManaged( &d_Ipiv, sizeof( U ) * N ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( d_Ipiv, sizeof( U ) * N, device, stream ) );
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

    /* step 4: LU factorization */
    if ( pivot_on ) {
        CUDA_RT_CALL( cusolverDnXgetrf( cusolverH,
                                        NULL,
                                        N,
                                        N,
                                        CUDA_R_64F,
                                        A,
                                        lda,
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
                                        N,
                                        N,
                                        CUDA_R_64F,
                                        A,
                                        lda,
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
    CUDA_RT_CALL( cudaMemAdvise( d_Ipiv, sizeof( U ) * N, cudaMemAdviseSetReadMostly, device ) );

    /*
     * step 5: solve A*X = B
     */

    if ( pivot_on ) {
        CUDA_RT_CALL( cusolverDnXgetrs( cusolverH,
                                        NULL,
                                        CUBLAS_OP_N,
                                        N,
                                        1, /* nrhs */
                                        CUDA_R_64F,
                                        A,
                                        lda,
                                        d_Ipiv,
                                        CUDA_R_64F,
                                        B,
                                        ldb,
                                        d_info ) );
    } else {
        CUDA_RT_CALL( cusolverDnXgetrs( cusolverH,
                                        NULL,
                                        CUBLAS_OP_N,
                                        N,
                                        1, /* nrhs */
                                        CUDA_R_64F,
                                        A,
                                        lda,
                                        nullptr,
                                        CUDA_R_64F,
                                        B,
                                        ldb,
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

    if ( d_Ipiv )
        CUDA_RT_CALL( cudaFree( d_Ipiv ) );
    if ( d_info )
        CUDA_RT_CALL( cudaFree( d_info ) );
    if ( bufferOnDevice )
        CUDA_RT_CALL( cudaFree( bufferOnDevice ) );
    if ( bufferOnHost )
        CUDA_RT_CALL( cudaFree( bufferOnHost ) );
    if ( cusolverH )
        CUDA_RT_CALL( cusolverDnDestroy( cusolverH ) );
    if ( stream )
        CUDA_RT_CALL( cudaStreamDestroy( stream ) );
}

int main( int argc, char *argv[] ) {

    int device = -1;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    const int64_t m { 39000 };
    const int64_t lda { m };
    const int64_t ldb { m };

    double *m_A {};
    double *m_B {};

    CUDA_RT_CALL( cudaMallocManaged( &m_A, sizeof( double ) * lda * m ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_B, sizeof( double ) * m ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( m_A, sizeof( double ) * lda * m, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( m_B, sizeof( double ) * m, device, NULL ) );

    // Generate random numbers on the GPU
    curandGenerator_t gen;
    CUDA_RT_CALL( curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT ) );
    CUDA_RT_CALL( curandSetPseudoRandomGeneratorSeed( gen, 1234ULL ) );
    std::printf(
        "Number generation of %lu values (A): %lu\n", static_cast<size_t>( m ) * lda, sizeof( double ) * lda * m );
    CUDA_RT_CALL( curandGenerateNormalDouble( gen, m_A, static_cast<size_t>( m ) * lda, 100.0, 50.0 ) );
    std::printf( "Number generation of %lu values (B): %lu\n", static_cast<size_t>( m ), sizeof( double ) * m );
    CUDA_RT_CALL( curandGenerateNormalDouble( gen, m_B, static_cast<size_t>( m ), 100.0, 50.0 ) );

    // Managed Memory
    std::printf( "Run LU Decomposition\n" );
    SingleGPUManaged( device, m, lda, ldb, m_A, m_B );

    CUDA_RT_CALL( cudaFree( m_A ) );
    CUDA_RT_CALL( cudaFree( m_B ) );

    return ( EXIT_SUCCESS );
}