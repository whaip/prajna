#include <boost/dll/shared_library.hpp>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "prajna/assert.hpp"
namespace prajna::jit {
// 对应 HIP 类型别名（用 uintptr_t 避免 HIP headers）
using hipDevice_t = int;          // 表示 GPU 设备 ID
using hipCtx_t = uintptr_t;       // 表示 HIP 运行时上下文指针
using hipModule_t = uintptr_t;    // 表示 HSACO 模块指针
using hipFunction_t = uintptr_t;  // 表示 GPU 内核函数指针

class HipRuntimeLoader {
   public:
    HipRuntimeLoader() : hip_available(false) {
        try {
// 根据平台加载 ROCm runtime
#ifdef _WIN32
            rocm_so = std::make_unique<boost::dll::shared_library>("amdhip64.dll",
                                                                   boost::dll::load_mode::rtld_lazy);
#else
            rocm_so = std::make_unique<boost::dll::shared_library>("/opt/rocm/lib/libamdhip64.so",
                                                                   boost::dll::load_mode::rtld_lazy);

#endif
            // 获取 HIP API 函数指针，绑定到对应的 std::function
            hipInit = rocm_so->get<int(int)>("hipInit");
            hipDeviceGet = rocm_so->get<int(hipDevice_t*, int)>("hipDeviceGet");
            hipCtxCreate = rocm_so->get<int(hipCtx_t*, unsigned int, hipDevice_t)>("hipCtxCreate");
            hipModuleLoad = rocm_so->get<int(hipModule_t*, const char*)>("hipModuleLoad");
            hipModuleGetFunction =
                rocm_so->get<int(hipFunction_t*, hipModule_t, const char*)>("hipModuleGetFunction");

            // 初始化 runtime 只做一次
            auto init_result = hipInit(0);
            if (init_result != 0) {
                std::cerr << "Warning: HIP initialization failed with error code " << init_result 
                          << ". GPU functionality will be disabled." << std::endl;
                return;
            }
            
            auto device_get_result = hipDeviceGet(&device, 0);
            if (device_get_result != 0) {
                std::cerr << "Warning: HIP device detection failed with error code " << device_get_result 
                          << ". GPU functionality will be disabled." << std::endl;
                return;
            }
            
            auto ctx_result = hipCtxCreate(&context, 0, device);
            if (ctx_result != 0) {
                std::cerr << "Warning: HIP context creation failed with error code " << ctx_result 
                          << ". GPU functionality will be disabled." << std::endl;
                return;
            }
            
            // 如果所有初始化都成功，设置可用标志
            hip_available = true;
            std::cout << "HIP initialized successfully. GPU functionality is enabled." << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to initialize HIP runtime: " << e.what() 
                      << ". GPU functionality will be disabled." << std::endl;
            hip_available = false;
        }
    }

    // 加载 HSACO 文件中的内核函数并返回其句柄
    hipFunction_t LoadKernelFunction(const std::string& hsaco_path,
                                     const std::string& kernel_name) {
        if (!hip_available) {
            std::cerr << "Error: HIP is not available. Cannot load kernel function: " << kernel_name << std::endl;
            return 0;  // 返回无效的函数句柄
        }

        hipModule_t module;
        // 加载 HSACO 文件到模块
        auto module_load_result = hipModuleLoad(&module, hsaco_path.c_str());
        if (module_load_result != 0) {
            std::cerr << "Error: Failed to load HSACO module from " << hsaco_path 
                      << " with error code " << module_load_result << std::endl;
            return 0;
        }

        hipFunction_t kernel_func;
        // 从模块中获取指定内核函数
        auto get_function_result = hipModuleGetFunction(&kernel_func, module, kernel_name.c_str());
        if (get_function_result != 0) {
            std::cerr << "Error: Failed to get kernel function " << kernel_name 
                      << " with error code " << get_function_result << std::endl;
            return 0;
        }

        return kernel_func;
    }

    // 检查 HIP 是否可用
    bool IsAvailable() const {
        return hip_available;
    }

   private:
    // HIP 可用性标志
    bool hip_available;
    
    // HIP 运行时库的智能指针，管理动态加载的 libamdhip64.so 或 amdhip64.dll
    std::unique_ptr<boost::dll::shared_library> rocm_so;

    // HIP runtime context
    hipDevice_t device;  // GPU 设备 ID
    hipCtx_t context;    // GPU 上下文句柄

    // HIP API 函数指针，用于动态调用
    std::function<int(int)> hipInit;  // 初始化 HIP 运行时
    std::function<int(hipDevice_t*, int)> hipDeviceGet;
    std::function<int(hipCtx_t*, unsigned int, hipDevice_t)> hipCtxCreate;  // 创建 GPU 上下文
    std::function<int(hipModule_t*, const char*)> hipModuleLoad;            // 加载 HSACO 模块
    std::function<int(hipFunction_t*, hipModule_t, const char*)> hipModuleGetFunction;
};
}  // namespace prajna::jit