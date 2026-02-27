local cuda_home = os.getenv("CUDA_HOME") or "/usr/local/cuda"
local cuda_arch = os.getenv("CUDA_ARCH") or "sm_75"
local compute_arch = cuda_arch:gsub("^sm_", "compute_")
local cuda_inc = path.join(cuda_home, "include")
local cuda_lib64 = path.join(cuda_home, "lib64")

-- Basic sanity checks (informative; won't hard-fail xmake but will warn)
if not os.isdir(cuda_home) then
    print("warning: CUDA_HOME not found at " .. tostring(cuda_home) .. ". Please set CUDA_HOME to your CUDA installation.")
end

if not os.isdir(cuda_inc) then
    print("warning: CUDA include dir not found: " .. tostring(cuda_inc))
end

-- Use xmake's builtin CUDA rule if available
-- Pass the CUDA_HOME as toolchain if supported
add_rules("cuda", {cuda_sdk = cuda_home})

--
target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_deps("llaisys-utils")
    add_defines("ENABLE_NVIDIA_API")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_includedirs(cuda_inc, "include", "../src/device")
    if is_plat("linux") then
        add_linkdirs(cuda_lib64)
    end

    add_links("cudart", "cublas", "cudadevrt")   -- 添加 cudadevrt

    local gencode = string.format("-gencode=arch=%s,code=%s", compute_arch, cuda_arch)
    add_cuflags(gencode, "-Xcompiler -fPIC", "-rdc=true", {force = true})  -- 启用 RDC

    add_files("../src/device/nvidia/*.cu")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_deps("llaisys-tensor")
    add_defines("ENABLE_NVIDIA_API")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_includedirs(cuda_inc, "include", "../src/ops", "../src")
    if is_plat("linux") then
        add_linkdirs(cuda_lib64)
    end

    add_links("cudart", "cublas", "cudadevrt")   -- 添加 cudadevrt

    local gencode = string.format("-gencode=arch=%s,code=%s", compute_arch, cuda_arch)
    add_cuflags(gencode, "-Xcompiler -fPIC", "-rdc=true", {force = true})  -- 启用 RDC

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()