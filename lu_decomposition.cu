#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include <curand.h>
#include <cusolverDn.h>
#include <cusolverMg.h>

#include "utilities.h"

#define VERIFY 0

constexpr int pivot_on { 1 };

void SingleGPUManaged( const size_t &N, const size_t &lda, const size_t &ldb, double *A, double *B, double *X ) {

    std::printf( "\ncuSolver: SingleGPUManaged GETRF\n" );

#if VERIFY
    double *B_input {};
    double *A_input {};

    CUDA_RT_CALL( cudaMallocManaged( &A_input, sizeof( double ) * lda * N ) );
    CUDA_RT_CALL( cudaMallocManaged( &B_input, sizeof( double ) * N ) );

    for ( int i = 0; i < N; i++ ) {
        B_input[i] = B[i];
    }

    for ( int i = 0; i < lda * N; i++ ) {
        A_input[i] = A[i];
    }
#endif

    // Start timer
    cudaEvent_t startEvent { nullptr };
    cudaEvent_t stopEvent { nullptr };
    float       elapsed_gpu_ms {};

    CUDA_RT_CALL( cudaEventCreate( &startEvent, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stopEvent, cudaEventBlockingSync ) );

    int lwork {}; /* size of workspace */

    int *d_Ipiv { nullptr }; /* pivoting sequence */
    int *d_info { nullptr }; /* error info */

    double *d_work { nullptr }; /* device workspace for getrf */

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
    CUDA_RT_CALL( cudaMallocManaged( &d_info, sizeof( int ) ) );

    if ( pivot_on ) {
        CUDA_RT_CALL( cudaMallocManaged( &d_Ipiv, sizeof( int ) * N ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( d_Ipiv, N * sizeof( int ), 0, stream ) );
    }

    CUDA_RT_CALL( cusolverDnDgetrf_bufferSize( cusolverH, N, N, A, lda, &lwork ) );

    CheckMemoryUsed( 1 );

    std::printf( "lwork = %d\n", lwork );
    std::printf( "\tAllocate device workspace, lwork = %lu\n", sizeof( double ) * lwork );

    CUDA_RT_CALL( cudaMallocManaged( &d_work, ( sizeof( double ) * lwork ) ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( d_work, sizeof( double ) * lwork, 0, stream ) );

    // Check GPU memory used on single GPU
    CheckMemoryUsed( 1 );

    CUDA_RT_CALL( cudaEventRecord( startEvent ) );

    /* step 4: LU factorization */
    if ( pivot_on ) {
        CUDA_RT_CALL( cusolverDnDgetrf( cusolverH, N, N, A, lda, d_work, d_Ipiv, d_info ) );
    } else {
        CUDA_RT_CALL( cusolverDnDgetrf( cusolverH, N, N, A, lda, d_work, nullptr, d_info ) );
    }

    // Must be here to retrieve d_info
    CUDA_RT_CALL( cudaStreamSynchronize( stream ) );

    if ( *d_info ) {
        throw std::runtime_error( std::to_string( -*d_info ) + "-th parameter is wrong (cusolverDnDgetrf) \n" );
    }

    /*
     * step 5: solve A*X = B
     */
    if ( pivot_on ) {
        CUDA_RT_CALL( cusolverDnDgetrs( cusolverH,
                                        CUBLAS_OP_N,
                                        N,
                                        1, /* nrhs */
                                        A,
                                        lda,
                                        d_Ipiv,
                                        B,
                                        ldb,
                                        d_info ) );
    } else {
        CUDA_RT_CALL( cusolverDnDgetrs( cusolverH,
                                        CUBLAS_OP_N,
                                        N,
                                        1, /* nrhs */
                                        A,
                                        lda,
                                        nullptr,
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

    if ( d_Ipiv )
        CUDA_RT_CALL( cudaFree( d_Ipiv ) );
    if ( d_info )
        CUDA_RT_CALL( cudaFree( d_info ) );
    if ( d_work )
        CUDA_RT_CALL( cudaFree( d_work ) );
    if ( cusolverH )
        CUDA_RT_CALL( cusolverDnDestroy( cusolverH ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( B, sizeof( double ) * N, cudaCpuDeviceId, stream ) );

#if VERIFY
    for ( int i = 0; i < N; i++ ) {
        X[i] = B[i];
    }

    // Calculate Residual Error
    CalculateResidualError( N, lda, A_input, B_input, X );
#endif
}

int main( int argc, char *argv[] ) {

    int device = -1;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    const size_t m { 39000 };
    const size_t lda { m };
    const size_t ldb { m };

    double *m_A {};
    double *m_B {};
    double *m_single_X {};

    CUDA_RT_CALL( cudaMallocManaged( &m_A, sizeof( double ) * lda * m ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_B, sizeof( double ) * m ) );
    CUDA_RT_CALL( cudaMallocManaged( &m_single_X, sizeof( double ) * m ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( m_A, sizeof( double ) * lda * m, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( m_B, sizeof( double ) * m, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( m_single_X, sizeof( double ) * m, cudaCpuDeviceId, NULL ) );
    // CUDA_RT_CALL( cudaMemPrefetchAsync( m_single_X, sizeof( double ) * m, device, NULL ) );

    // Generate random numbers on the GPU
    curandGenerator_t gen;
    CUDA_RT_CALL( curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT ) );
    CUDA_RT_CALL( curandSetPseudoRandomGeneratorSeed( gen, 1234ULL ) );
    std::printf( "Number generation of %lu values (A): %lu\n", lda * m, sizeof( double ) * lda * m );
    CUDA_RT_CALL( curandGenerateNormalDouble( gen, m_A, lda * m, 100.0, 50.0 ) );
    std::printf( "Number generation of %lu values (B): %lu\n", m, sizeof( double ) * m );
    CUDA_RT_CALL( curandGenerateNormalDouble( gen, m_B, m, 100.0, 50.0 ) );

    // Managed Memory
    std::printf( "Run LU Decomposition\n" );
    SingleGPUManaged( m, lda, ldb, m_A, m_B, m_single_X );

    CUDA_RT_CALL( cudaFree( m_A ) );
    CUDA_RT_CALL( cudaFree( m_B ) );
    CUDA_RT_CALL( cudaFree( m_single_X ) );

    return ( EXIT_SUCCESS );
}