// File       : gemm.ispc
// Description: ISPC GEMM body
// Copyright 2020 ETH Zurich. All Rights Reserved.

#define __BLOCK__ programCount

export void sgemv_ispc(
    const uniform float * uniform A, 
    const uniform float * uniform x, 
    float * uniform y, 
    const uniform size_t N){

    
    // Wrong solution: parallel write to uniform value in y
    // for(uniform int I = 0; I < N; I++){
    //     foreach(J = 0 ... N)
    //         y[I] += A[I*N+J] * x[J];
    // }

    // Suboptimal solution: Columnwise access in matrix A
    // foreach(I=0 ... N){
    //     for(uniform int J = 0; J < N; J++){
    //         y[I] += A[I*N+J]*x[J];
    //     }
    // }

    // Optimal solution
    uniform float tmp[__BLOCK__];

    for(uniform int I = 0; I < N; I++){
        foreach(j = 0 ... __BLOCK__)
            tmp[j] = 0.0f;

        for(uniform int J = 0; J < N; J+= __BLOCK__)
            foreach(j= 0 ... __BLOCK__)
                tmp[j] += A[I*N+J+j] * x[J+j];

        for(uniform int j=0; j <__BLOCK__;j++)
            y[I] += tmp[j];

    }


};