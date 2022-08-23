// File       : gemm.ispc
// Description: ISPC GEMM body
// Copyright 2020 ETH Zurich. All Rights Reserved.

#define __BLOCK__ programCount

export void sgemv_ispc(
    const uniform float * uniform A, 
    const uniform float * uniform x, 
    float * uniform y, 
    const uniform size_t N){

    uniform float tmp[__BLOCK__];


};