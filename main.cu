#include <optix.h>
#include <iostream>
#include <cstring>

// Error checking macro for CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl; \
            return 1; \
        } \
    } while (0)

int main() {
    // Initialize OptiX library
    if (optixInit() != OPTIX_SUCCESS) {
        std::cerr << "Failed to initialize OptiX." << std::endl;
        return 1;
    }

    // Set up device context
    CUcontext cuCtx = 0;
    OptixDeviceContext optixContext;
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr;
    options.logCallbackLevel = 4;

    if (optixDeviceContextCreate(cuCtx, &options, &optixContext) != OPTIX_SUCCESS) {
        std::cerr << "Failed to create OptiX context." << std::endl;
        return 1;
    }

    // Load PTX code for ray generation shader
    const char* ptxCode = R"(
        .version 7.0
        .target sm_70
        .address_size 64

        // Ray generation program
        .entry __raygen__simple {
            // For simplicity, empty shader
        }
    )";

    // Compile and load module
    OptixModule module;
    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    if (optixModuleCreateFromPTX(optixContext, &module_compile_options, &pipeline_compile_options,
                                 ptxCode, strlen(ptxCode), nullptr, 0, &module) != OPTIX_SUCCESS) {
        std::cerr << "Failed to create module." << std::endl;
        return 1;
    }

    // Create ray generation program group
    OptixProgramGroup raygen_prog_group;
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__simple";

    if (optixProgramGroupCreate(optixContext, &raygen_prog_group_desc, 1, &program_group_options,
                                nullptr, nullptr, &raygen_prog_group) != OPTIX_SUCCESS) {
        std::cerr << "Failed to create program group." << std::endl;
        return 1;
    }

    // Pipeline creation
    OptixPipeline pipeline;
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
    
    if (optixPipelineCreate(optixContext, &pipeline_compile_options, &pipeline_link_options,
                            &raygen_prog_group, 1, nullptr, nullptr, &pipeline) != OPTIX_SUCCESS) {
        std::cerr << "Failed to create pipeline." << std::endl;
        return 1;
    }

    std::cout << "OptiX pipeline created successfully!" << std::endl;

    // Clean up
    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(raygen_prog_group);
    optixModuleDestroy(module);
    optixDeviceContextDestroy(optixContext);

    return 0;
}
