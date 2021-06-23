# cusolver_examples

### cuSOLVER (Single GPU)
Running `./lu_decomposition_cusolver {size} {loops} {algo}`

where `algo` is either 0 - New or 1 - Legacy  --- https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXgetrf

### cuSOLVERMg (MultiGPU)
Running `./lu_decomposition_cusolvermg {size} {loops} {gpus}`

### MAGMA (Single GPU || MultiGPU)
Running `./lu_decomposition_magma {size} {loops} {gpus}`
