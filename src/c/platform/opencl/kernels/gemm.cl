/**
 * Venus Inference Engine - GEMM OpenCL Kernels
 * 
 * Tiled matrix multiplication optimized for various GPU architectures.
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

#ifndef TILE_M
#define TILE_M 8
#endif
#ifndef TILE_N
#define TILE_N 8
#endif
#ifndef TILE_K
#define TILE_K 8
#endif
#ifndef LOCAL_MEM_PAD
#define LOCAL_MEM_PAD 1
#endif
#ifndef DEQUANT_BLOCK
#define DEQUANT_BLOCK 32
#endif

// Tiled GEMM: C = alpha * A @ B + beta * C
__kernel void gemm_f32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row >= M || col >= N) return;
    
    __local float tileA[TILE_M][TILE_K + LOCAL_MEM_PAD];
    __local float tileB[TILE_K][TILE_N + LOCAL_MEM_PAD];
    
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    const int tileRow = get_group_id(0) * TILE_M;
    const int tileCol = get_group_id(1) * TILE_N;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Load tiles cooperatively
        int aRow = tileRow + localRow;
        int aCol = t * TILE_K + localCol;
        int bRow = t * TILE_K + localRow;
        int bCol = tileCol + localCol;
        
        if (aRow < M && aCol < K)
            tileA[localRow][localCol] = A[aRow * K + aCol];
        else
            tileA[localRow][localCol] = 0.0f;
        
        if (bRow < K && bCol < N)
            tileB[localRow][localCol] = B[bRow * N + bCol];
        else
            tileB[localRow][localCol] = 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Apply alpha/beta scaling
    if (beta != 0.0f) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    } else {
        C[row * N + col] = alpha * sum;
    }
}

// GEMM with Q8 quantized B matrix (dequantize on-the-fly)
__kernel void gemm_q8(
    __global const float* A,
    __global const char* B_q8,
    __global const float* scales,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Process in blocks of 32 (Q8_0 block size)
    for (int k = 0; k < K; k += DEQUANT_BLOCK) {
        int block_idx = (col * K + k) / DEQUANT_BLOCK;
        float scale = scales[block_idx];
        
        #pragma unroll 8
        for (int i = 0; i < DEQUANT_BLOCK && (k + i) < K; i++) {
            float a_val = A[row * K + k + i];
            float b_val = (float)B_q8[col * K + k + i] * scale;
            sum += a_val * b_val;
        }
    }
    
    C[row * N + col] = sum;
}

// GEMM with Q4 quantized B matrix (packed 2 values per byte)
__kernel void gemm_q4(
    __global const float* A,
    __global const uchar* B_q4,
    __global const float* scales,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Process in blocks of 32 (Q4_0 block size)
    for (int k = 0; k < K; k += DEQUANT_BLOCK) {
        int block_idx = (col * K + k) / DEQUANT_BLOCK;
        float scale = scales[block_idx];
        
        for (int i = 0; i < DEQUANT_BLOCK && (k + i) < K; i += 2) {
            int byte_idx = (col * K + k + i) / 2;
            uchar packed = B_q4[byte_idx];
            
            // Unpack two 4-bit values (signed, centered at 8)
            int v0 = (int)(packed & 0x0F) - 8;
            int v1 = (int)(packed >> 4) - 8;
            
            float a0 = A[row * K + k + i];
            float a1 = (k + i + 1 < K) ? A[row * K + k + i + 1] : 0.0f;
            
            sum += a0 * (float)v0 * scale;
            sum += a1 * (float)v1 * scale;
        }
    }
    
    C[row * N + col] = sum;
}

