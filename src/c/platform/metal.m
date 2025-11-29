#include "platform.h"

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>

// Forward declarations for Metal-specific structures
typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;
} MetalBackend;

static MetalBackend* g_metal_backend = NULL;

void init_metal(void) {
    printf("Initializing Metal backend...\n");
    g_metal_backend = (MetalBackend*)calloc(1, sizeof(MetalBackend));

    g_metal_backend->device = MTLCreateSystemDefaultDevice();
    if (!g_metal_backend->device) {
        printf("Metal is not supported on this device.\n");
        free(g_metal_backend);
        g_metal_backend = NULL;
        return;
    }

    g_metal_backend->command_queue = [g_metal_backend->device newCommandQueue];
    
    printf("Metal backend initialized successfully for device: %s\n", [g_metal_backend->device.name UTF8String]);
}

void cleanup_metal(void) {
    if (g_metal_backend) {
        // Release Metal objects here
        NSLog(@"Cleaning up Metal backend.");
        free(g_metal_backend);
    }
}

// GEMM implementation using Metal Performance Shaders
void metal_gemm_f32(const float* a, const float* b, float* c,
                    int m, int n, int k, float alpha, float beta) {
    
    if (!g_metal_backend || !g_metal_backend->device) {
        printf("Metal backend not initialized. Falling back to CPU GEMM.\n");
        // Fallback to a CPU implementation if Metal is not available
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, a, k, b, n, beta, c, n);
        return;
    }

    @autoreleasepool {
        // Create MTLBuffers for the matrices
        id<MTLBuffer> bufferA = [g_metal_backend->device newBufferWithBytes:a length:m * k * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [g_metal_backend->device newBufferWithBytes:b length:k * n * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [g_metal_backend->device newBufferWithBytes:c length:m * n * sizeof(float) options:MTLResourceStorageModeShared];

        // Create a command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [g_metal_backend->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> compute_encoder = [commandBuffer computeCommandEncoder];

        // Create matrix descriptors
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];

        // Create matrix objects
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        // Create the GEMM kernel
        MPSMatrixMultiplication *gemmKernel = [[MPSMatrixMultiplication alloc] initWithDevice:g_metal_backend->device transposeLeft:NO transposeRight:NO resultRows:m resultColumns:n interiorColumns:k alpha:alpha beta:beta];

        // Encode the kernel to the command buffer
        [gemmKernel encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];

        // Execute the command buffer
        [compute_encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy data back from bufferC to c
        memcpy(c, bufferC.contents, m * n * sizeof(float));
    }
}

#endif // __APPLE__
