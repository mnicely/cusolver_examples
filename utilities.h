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

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <random>
#include <vector>

 #define FMULS_GETRF(m_, n_) ( ((m_) < (n_)) \
     ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_) - 1. ) + (n_)) + (2. / 3.) * (m_)) \
     : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_) - 1. ) + (m_)) + (2. / 3.) * (n_)) )
 #define FADDS_GETRF(m_, n_) ( ((m_) < (n_)) \
     ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_)      ) - (n_)) + (1. / 6.) * (m_)) \
     : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_)      ) - (m_)) + (1. / 6.) * (n_)) )

 #define FLOPS_ZGETRF(m_, n_) (6. * FMULS_GETRF((double)(m_), (double)(n_)) + 2.0 * FADDS_GETRF((double)(m_), (double)(n_)) )

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

/**************************************************/
/* RANDOM NUMBERS GENERATION STRUCTS AND FUNCTION */
/**************************************************/
template<typename T>
struct Rand {
    std::default_random_engine  gen;
    std::normal_distribution<T> dist;
    Rand( ) : gen {}, dist( 100.0, 50.0 ) {}

    __host__ T operator( )( const T &VecElem ) {
        return ( dist( gen ) );
    }
};

constexpr double tolerance { 1e-6 };

template<typename T>
constexpr T Max( const T &a, const T &b ) {
    return ( ( a > b ) ? ( a ) : ( b ) );
}

template<typename T>
constexpr T Min( const T &a, const T &b ) {
    return ( ( a < b ) ? ( a ) : ( b ) );
}

template<typename T>
constexpr T Idx1f( const T &i ) {
    return ( i - 1 );
}

template<typename T>
constexpr int64_t Idx2f( const T &i, const T &j, const T &lda ) {
    return ( ( ( j - 1 ) * static_cast<int64_t>( lda ) ) + ( i - 1 ) );
}

void GetDeviceProperties( const int &num_devices, int *device_list ) {

    std::printf( "There are %d GPUs\n", num_devices );
    for ( int j = 0; j < num_devices; j++ ) {
        device_list[j] = j;
        cudaDeviceProp prop;
        CUDA_RT_CALL( cudaGetDeviceProperties( &prop, j ) );
        std::printf( "\tDevice %d, %s, cc %d.%d \n", j, prop.name, prop.major, prop.minor );
    }
}

void CheckMemoryUsed( const int &num_devices ) {
    // Check how much memory is being used across all devices
    size_t mem_free {};
    size_t mem_total {};
    size_t mem_used {};

    int currentDev {}; /* record current device ID */
    CUDA_RT_CALL( cudaGetDevice( &currentDev ) );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    for ( int idx = 0; idx < num_devices; idx++ ) {

        CUDA_RT_CALL( cudaSetDevice( idx ) );

        CUDA_RT_CALL( cudaMemGetInfo( &mem_free, &mem_total ) );
        std::printf( "Memory used on device %d: %lu\n", idx, ( mem_total - mem_free ) );
        mem_used += ( mem_total - mem_free );
    }

    CUDA_RT_CALL( cudaSetDevice( currentDev ) );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    std::printf( "Total memory used: %lu\n", mem_used );
}

template<typename T>
void CreateRandomData( const std::string &str, const int64_t &size, T *D ) {

    const size_t size_bytes { sizeof( T ) * size };

    std::printf( "Number generation of %lu values (%s): %lu (bytes)\n", size, str.c_str( ), size_bytes );

    curandGenerator_t gen;
    CUDA_RT_CALL( curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT ) );
    CUDA_RT_CALL( curandSetPseudoRandomGeneratorSeed( gen, 1234ULL ) );
    CUDA_RT_CALL( curandGenerateNormalDouble( gen, D, size, 100.0, 50.0 ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( D, size_bytes, cudaCpuDeviceId, NULL ) );
}

/* compute |x|_inf */
template<typename T>
T VecNrmInf( const int &N, const T *Z ) {

    T max_nrm {};
    for ( int row = 1; row <= N; row++ ) {

        T xi { Z[Idx1f( row )] };
        max_nrm = ( max_nrm > std::fabs( xi ) ) ? max_nrm : std::fabs( xi );
    }

    return ( max_nrm );
}

template<typename T, typename U>
void CalculateResidualError( const U &N, const U &lda, const T *A, const T *B, const T *X ) {

    std::printf( "Measure residual error |b - A*x|\n" );
    T max_err {};
    for ( U row = 1; row <= N; row++ ) {

        T sum {};
        for ( U col = 1; col <= N; col++ ) {

            T Aij { A[Idx2f( row, col, lda )] };
            T xj { X[Idx1f( col )] };
            sum += Aij * xj;
        }
        T bi { B[Idx1f( row )] };
        T err { std::fabs( bi - sum ) };

        max_err = ( max_err > err ) ? max_err : err;
    }

    T x_nrm_inf { VecNrmInf( N, X ) };
    T b_nrm_inf { VecNrmInf( N, B ) };

    T A_nrm_inf { 4.0 };
    T rel_err { max_err / ( A_nrm_inf * x_nrm_inf + b_nrm_inf ) };

    std::printf( "\n|b - A*x|_inf = %E\n", max_err );
    std::printf( "|x|_inf = %E\n", x_nrm_inf );
    std::printf( "|b|_inf = %E\n", b_nrm_inf );
    std::printf( "|A|_inf = %E\n", A_nrm_inf );
    /* relative error is around machine zero  */
    /* the user can use |b - A*x|/(N*|A|*|x|+|b|) as well */
    std::printf( "|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err );
}

template<typename T>
void WorkspaceFree( const int &num_devices,
                    const int *deviceIdA,   /* <int> dimension num_devices */
                    T **       array_d_work /* <t> num_devices, host array */
                                            /* array_d_work[j] points to device workspace of device j */
) {
    int currentDev {}; /* record current device ID */
    CUDA_RT_CALL( cudaGetDevice( &currentDev ) );

    for ( int idx = 0; idx < num_devices; idx++ ) {

        int deviceId { deviceIdA[idx] };

        /* WARNING: we need to set device before any runtime API */
        CUDA_RT_CALL( cudaSetDevice( deviceId ) );

        if ( array_d_work[idx] ) {
            CUDA_RT_CALL( cudaFree( array_d_work[idx] ) );
        }
    }
    CUDA_RT_CALL( cudaSetDevice( currentDev ) );
}

template<typename T>
void WorkspaceAlloc( const int &   num_devices,
                     const int *   deviceIdA,   /* <int> dimension num_devices */
                     const size_t &sizeInBytes, /* number of bytes per device */
                     T **          array_d_work /* <t> num_devices, host array */
                                                /* array_d_work[j] points to device workspace of device j */
) {

    int currentDev {}; /* record current device ID */
    CUDA_RT_CALL( cudaGetDevice( &currentDev ) );

    for ( int idx = 0; idx < num_devices; idx++ ) {

        int deviceId { deviceIdA[idx] };
        T * d_workspace {};

        /* WARNING: we need to set device before any runtime API */
        CUDA_RT_CALL( cudaSetDevice( deviceId ) );

        CUDA_RT_CALL( cudaMallocManaged( &d_workspace, sizeInBytes ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync( d_workspace, sizeInBytes, deviceId, NULL ) );
        // CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_workspace ), sizeInBytes ) );

        array_d_work[idx] = d_workspace;
    }
    CUDA_RT_CALL( cudaSetDevice( currentDev ) );
}

/*
 * Z is an arbitrary input
 * create a empty matrix Z with Z := 0
 */
template<typename T>
void CreateMat( const int &num_devices,
                const int *deviceIdZ, /* <int> dimension num_devices */
                const int &N_Z,       /* number of columns of global Z */
                const int &T_Z,       /* number of columns per column tile */
                const int &LLD_Z,     /* leading dimension of local Z */
                T **       array_d_Z  /* host pointer array of dimension num_devices */
) {

    int currentDev {}; /* record current device id */

    CUDA_RT_CALL( cudaGetDevice( &currentDev ) );

    const int Z_num_blks { ( N_Z + T_Z - 1 ) / T_Z };
    const int max_Z_num_blks_per_device { ( Z_num_blks + num_devices - 1 ) / num_devices };

    for ( int p = 0; p < num_devices; p++ ) {

        CUDA_RT_CALL( cudaSetDevice( deviceIdZ[p] ) );

        // std::printf( "Allocating %lu on device %d\n", sizeof( T ) * LLD_Z * T_Z * max_Z_num_blks_per_device, p );

        /* Allocate max_A_num_blks_per_device blocks per device */
        // CUDA_RT_CALL( cudaMalloc( &( array_d_Z[p] ), sizeof( T ) * LLD_Z * T_Z * max_Z_num_blks_per_device ) );
        CUDA_RT_CALL( cudaMallocManaged( &( array_d_Z[p] ), sizeof( T ) * LLD_Z * T_Z * max_Z_num_blks_per_device ) );
        CUDA_RT_CALL( cudaMemPrefetchAsync(
            array_d_Z[p], sizeof( T ) * LLD_Z * T_Z * max_Z_num_blks_per_device, deviceIdZ[p], NULL ) );

        /* A := 0 */
        CUDA_RT_CALL( cudaMemset( array_d_Z[p], 0, sizeof( T ) * LLD_Z * T_Z * max_Z_num_blks_per_device ) );
    }

    CUDA_RT_CALL( cudaSetDevice( currentDev ) );
}

/*
 * Z is an arbitrary input
 */
template<typename T>
void DestroyMat( const int &num_devices,
                 const int *deviceIdZ, /* <int> dimension num_devices */
                 const int &N_Z,       /* number of columns of global Z */
                 const int &T_Z,       /* number of columns per column tile */
                 T **       array_d_Z )       /* host pointer array of dimension num_devices */
{

    int currentDev {}; /* record current device id */

    CUDA_RT_CALL( cudaGetDevice( &currentDev ) );

    for ( int p = 0; p < num_devices; p++ ) {

        CUDA_RT_CALL( cudaSetDevice( deviceIdZ[p] ) );

        if ( array_d_Z[p] ) {
            CUDA_RT_CALL( cudaFree( array_d_Z[p] ) );
        }
    }

    CUDA_RT_CALL( cudaSetDevice( currentDev ) );
}

void EnablePeerAccess( const int &num_devices ) {

    int currentDevice {};
    CUDA_RT_CALL( cudaGetDevice( &currentDevice ) );

    /* Remark: access granted by this cudaDeviceEnablePeerAccess is
     * unidirectional */
    /* Rows and columns represents a connectivity matrix between GPUs in the
     * system */
    for ( int row = 0; row < num_devices; row++ ) {

        CUDA_RT_CALL( cudaSetDevice( row ) );

        for ( int col = 0; col < num_devices; col++ ) {

            if ( row != col ) {
                int canAccessPeer {};
                CUDA_RT_CALL( cudaDeviceCanAccessPeer( &canAccessPeer, row, col ) );
                if ( canAccessPeer ) {
                    // std::printf( "\tEnable peer access from gpu %d to gpu % d\n", row, col );
                    CUDA_RT_CALL( cudaDeviceEnablePeerAccess( col, 0 ) );
                }
            }
        }
    }
    CUDA_RT_CALL( cudaSetDevice( currentDevice ) );
}

/*
 * Z is an arbitrary input
 */
template<typename T>
void MatPack2Unpack( const int &num_devices,
                     const int &N_Z,              /* number of columns of global Z */
                     const int &T_Z,              /* number of columns per column tile */
                     const int &LLD_Z,            /* leading dimension of local Z */
                     T **       array_d_Z_packed, /* host pointer array of dimension num_devices */
                     /* output */
                     T **array_d_Z_unpacked /* host pointer array of dimension num_blks */
) {

    const int num_blks { ( N_Z + T_Z - 1 ) / T_Z };

    for ( int p_z = 0; p_z < num_devices; p_z++ ) {

        T * d_Z { array_d_Z_packed[p_z] };
        int nz_blks {};

        for ( int JZ_blk_id = p_z; JZ_blk_id < num_blks; JZ_blk_id += num_devices ) {

            array_d_Z_unpacked[JZ_blk_id] = d_Z + ( size_t )LLD_Z * T_Z * nz_blks;
            nz_blks++;
        }
    }
}

/*
 * Z(IZ:IZ+M-1, JZ:JZ+N-1) := Y(1:M, 1:N)
 * Z, Y are arbitrary inputs
 */
template<typename T>
void MemcpyH2D( const int &num_devices,
                const int *deviceIdZ, /* <int> dimension num_devices */
                const int &M,         /* number of rows in local Z */
                const int &N,         /* number of columns in local Z */
                /* input */
                const T *  h_Z, /* host array, h_X is M-by-N with leading dimension ldb  */
                const int &ldz,
                /* output */
                const int &N_Z,              /* number of columns of global Z */
                const int &T_Z,              /* number of columns per column tile */
                const int &LLD_Z,            /* leading dimension of local Z */
                T **       array_d_Z_packed, /* host pointer array of dimension num_devices */
                const int &IZ,               /* base-1 */
                const int &JZ                /* base-1 */
) {
    /*  Quick return if possible */
    if ( ( 0 >= M ) || ( 0 >= N ) ) {
        throw std::runtime_error( "0 >= M or 0 >= N" );
    }

    /* consistent checking */
    if ( ldz < M ) {
        throw std::runtime_error( "ldz < M" );
    }

    int currentDev {}; /* record current device id */

    CUDA_RT_CALL( cudaGetDevice( &currentDev ) );

    const int num_blks { ( N_Z + T_Z - 1 ) / T_Z };

    std::vector<T *> array_d_Z_unpacked( num_blks );

    MatPack2Unpack( num_devices,
                    N_Z,              /* number of columns of global Z */
                    T_Z,              /* number of columns per column tile */
                    LLD_Z,            /* leading dimension of local Z */
                    array_d_Z_packed, /* host pointer array of size num_devices */
                    /* output */
                    array_d_Z_unpacked.data( ) /* host pointer arrya of size num_blks */
    );

    /* region of interest is Z(IZ:IZ+N-1, JZ:JZ+N-1) */
    const int N_hat { ( JZ - 1 ) + N }; /* JZ is base-1 */
    const int JZ_start_blk_id { ( JZ - 1 ) / T_Z };
    const int JZ_end_blk_id { ( N_hat - 1 ) / T_Z };

    for ( int p_z = 0; p_z < num_devices; p_z++ ) {

        CUDA_RT_CALL( cudaSetDevice( deviceIdZ[p_z] ) );

        /* region of interest: JZ_start_blk_id:1:JZ_end_blk_id */
        for ( int JZ_blk_id = p_z; JZ_blk_id <= JZ_end_blk_id; JZ_blk_id += num_devices ) {

            if ( JZ_blk_id < JZ_start_blk_id ) {
                continue;
            }

            /*
             * process column block of Z
             *       Z(Z_start_row:M_Z, Z_start_col : (Z_start_col + IT_Z-1) )
             */
            const int IBX_Z { ( 1 + JZ_blk_id * T_Z ) }; /* base-1 */
            const int Z_start_col { Max( JZ, IBX_Z ) };  /* base-1 */
            const int Z_start_row { IZ };                /* base-1 */

            const int bdd { Min( N_hat, ( IBX_Z + T_Z - 1 ) ) };
            const int IT_Z { Min( T_Z, ( bdd - Z_start_col + 1 ) ) };

            const int loc_Z_start_row { Z_start_row };                 /* base-1 */
            const int loc_Z_start_col { ( Z_start_col - IBX_Z ) + 1 }; /* base-1 */

            T *d_Z { array_d_Z_unpacked[JZ_blk_id] + Idx2f( loc_Z_start_row, loc_Z_start_col, LLD_Z ) };

            const T *h_ZZ { h_Z + Idx2f( Z_start_row - IZ + 1, Z_start_col - JZ + 1, ldz ) };

            CUDA_RT_CALL( cudaMemcpy2D( d_Z, /* dst */
                                        static_cast<size_t>( LLD_Z ) * sizeof( T ),
                                        h_ZZ, /* src */
                                        static_cast<size_t>( ldz ) * sizeof( T ),
                                        static_cast<size_t>( M ) * sizeof( T ),
                                        static_cast<size_t>( IT_Z ),
                                        cudaMemcpyHostToDevice ) );

        } /* for each tile per device */
    }     /* for each device */
    CUDA_RT_CALL( cudaSetDevice( currentDev ) );
}

/*
 * Y(1:M, 1:N) := Z(IZ:IZ+M-1, JZ:JZ+N-1)
 * Z, Y are arbitrary inputs
 */
template<typename T>
void MemcpyD2H( const int &num_devices,
                const int *deviceIdZ, /* <int> dimension num_devices */
                const int &M,         /* number of rows in local Z */
                const int &N,         /* number of columns in local Z */
                /* input */
                const int &N_Z,              /* number of columns of global Z */
                const int &T_Z,              /* number of columns per column tile */
                const int &LLD_Z,            /* leading dimension of local Z */
                T **       array_d_Z_packed, /* host pointer array of dimension num_devices */
                const int &IZ,               /* base-1 */
                const int &JZ,               /* base-1 */
                /* output */
                T *        h_Z, /* host array, h_Z is M-by-N with leading dimension ldz  */
                const int &ldz ) {

    int currentDev {}; /* record current device id */

    /*  Quick return if possible */
    if ( ( 0 >= M ) || ( 0 >= N ) ) {
        throw std::runtime_error( "0 >= M or 0 >= N" );
    }

    /* consistent checking */
    if ( ldz < M ) {
        throw std::runtime_error( "ldz < M" );
    }

    CUDA_RT_CALL( cudaGetDevice( &currentDev ) );

    const int num_blks { ( N_Z + T_Z - 1 ) / T_Z };

    std::vector<T *> array_d_Z_unpacked( num_blks );

    MatPack2Unpack( num_devices,
                    N_Z,              /* number of columns of global A */
                    T_Z,              /* number of columns per column tile */
                    LLD_Z,            /* leading dimension of local A */
                    array_d_Z_packed, /* host pointer array of size num_devices */
                    /* output */
                    array_d_Z_unpacked.data( ) /* host pointer arrya of size num_blks */
    );

    /* region of interest is Z(IZ:IZ+N-1, JZ:JZ+N-1) */
    const int N_hat { ( JZ - 1 ) + N }; /* JZ is base-1 */
    const int JZ_start_blk_id { ( JZ - 1 ) / T_Z };
    const int JZ_end_blk_id { ( N_hat - 1 ) / T_Z };

    for ( int p_z = 0; p_z < num_devices; p_z++ ) {

        CUDA_RT_CALL( cudaSetDevice( deviceIdZ[p_z] ) );

        /* region of interest: JA_start_blk_id:1:JA_end_blk_id */
        for ( int JZ_blk_id = p_z; JZ_blk_id <= JZ_end_blk_id; JZ_blk_id += num_devices ) {

            if ( JZ_blk_id < JZ_start_blk_id ) {
                continue;
            }

            /* process column block, Z(Z_start_row:M_Z, Z_start_col :
             * (Z_start_col + IT_Z-1) ) */
            const int IBX_Z { ( 1 + JZ_blk_id * T_Z ) }; /* base-1 */
            const int Z_start_col { Max( JZ, IBX_Z ) };  /* base-1 */
            const int Z_start_row { IZ };                /* base-1 */
            const int bdd { Min( N_hat, ( IBX_Z + T_Z - 1 ) ) };
            const int IT_Z { Min( T_Z, ( bdd - Z_start_col + 1 ) ) };
            const int loc_Z_start_row { Z_start_row };                 /* base-1 */
            const int loc_Z_start_col { ( Z_start_col - IBX_Z ) + 1 }; /* base-1 */

            const T *d_Z { array_d_Z_unpacked[JZ_blk_id] + Idx2f( loc_Z_start_row, loc_Z_start_col, LLD_Z ) };

            T *h_ZZ { h_Z + Idx2f( Z_start_row - IZ + 1, Z_start_col - JZ + 1, ldz ) };

            CUDA_RT_CALL( cudaMemcpy2D( h_ZZ, /* dst */
                                        static_cast<size_t>( ldz ) * sizeof( T ),
                                        d_Z, /* src */
                                        static_cast<size_t>( LLD_Z ) * sizeof( T ),
                                        static_cast<size_t>( M ) * sizeof( T ),
                                        static_cast<size_t>( IT_Z ),
                                        cudaMemcpyDeviceToHost ) );

        } /* for each tile per device */
    }     /* for each device */

    CUDA_RT_CALL( cudaSetDevice( currentDev ) );
}

#endif /* UTILITIES_H_ */