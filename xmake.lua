add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- 添加包仓库
add_requires("openblas", {configs = {
    shared = true, 
    pic = true,
    threads = true,      -- 启用线程
    openmp = true,       -- 启用 OpenMP
    avx2 = true,         -- 启用 AVX2 优化
    lapack = false       -- 如果你不需要 LAPACK 可以设为 false
}})

-- OpenBLAS 选项
option("openblas")
    set_default(true) 
    set_showmenu(true)
    set_description("Enable OpenBLAS for optimized linear algebra operations")
option_end()


-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

-- 全局编译选项
if is_plat("windows") then
    -- Windows平台配置
    add_cxflags("/arch:AVX2", "/O2", "/fp:fast", {force = true}) 
    add_cxflags("/openmp", {force = true})
    add_defines("ENABLE_F16C")
else
    -- Linux/Mac平台配置
    add_cxflags("-O3", "-march=native", "-mavx2", "-mfma", "-mf16c", {force = true})
    add_cxflags("-fopenmp", {force = true})
    add_ldflags("-fopenmp", {force = true})
end

-- 检查并安装所需的包
if has_config("openblas") then
    add_requires("openblas", {configs = {shared = true}})
    add_packages("openblas")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/cpu/*.cpp")
    
    -- 为CPU ops添加OpenBLAS支持
    if has_config("openblas") then
        add_defines("USE_OPENBLAS")
        
        if is_plat("windows") then
            local vcpkg_root = "E:/ai_/vcpkg"
            local vcpkg_triplet = "x64-windows"
            local vcpkg_installed = path.join(vcpkg_root, "installed", vcpkg_triplet)
            
            print("VCPKG installed path: " .. vcpkg_installed)
            
            local include_path = path.join(vcpkg_installed, "include")
            if os.isdir(include_path) then
                add_includedirs(include_path)
                print("Added include path: " .. include_path)
            end
            
            local lib_path = path.join(vcpkg_installed, "lib")
            if os.isdir(lib_path) then
                add_linkdirs(lib_path)
                print("Added lib path: " .. lib_path)
                
                -- 验证 openblas.lib 存在
                if os.isfile(path.join(lib_path, "openblas.lib")) then
                    print("Found openblas.lib")
                end
            end
            
            add_links("openblas")
            print("Linked to OpenBLAS")
        else
            add_packages("openblas")
        end
    end
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")
    
    if has_config("openblas") then
        add_defines("USE_OPENBLAS")
        add_packages("openblas")
    end

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    
    add_files("src/llaisys/*.cc")
    add_files("src/models/**.cpp")
    add_includedirs("src/models")
    set_installdir(".")
    
    if has_config("openblas") then
        if is_plat("windows") then
            local vcpkg_root = "E:/ai_/vcpkg"
            local vcpkg_triplet = "x64-windows"
            local vcpkg_installed = path.join(vcpkg_root, "installed", vcpkg_triplet)
            
            local lib_path = path.join(vcpkg_installed, "lib")
            if os.isdir(lib_path) then
                add_linkdirs(lib_path)
            end
            
            add_links("openblas")
            add_defines("USE_OPENBLAS")
        else
            add_packages("openblas")
            add_defines("USE_OPENBLAS")
        end
    end
    
    after_install(function (target)
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        os.mkdir("python/llaisys/libllaisys/")
        
        if is_plat("windows") then
            os.cp("build/windows/x64/release/*.dll", "python/llaisys/libllaisys/")
            os.cp("build/windows/x64/release/*.lib", "python/llaisys/libllaisys/")
            
            local vcpkg_root = "E:/ai_/vcpkg"
            local vcpkg_bin = path.join(vcpkg_root, "installed/x64-windows/bin")
            
            print("Looking for OpenBLAS DLL in: " .. vcpkg_bin)
            
            if os.isdir(vcpkg_bin) then
                local openblas_dlls = os.files(path.join(vcpkg_bin, "*.dll"))
                local copied = false
                
                for _, dll in ipairs(openblas_dlls) do
                    if dll:find("libopenblas") then
                        os.cp(dll, "python/llaisys/libllaisys/")
                        print("Copied OpenBLAS DLL: " .. dll)
                        copied = true
                    end
                end
                
                if not copied then
                    local possible_names = {"libopenblas.dll", "openblas.dll"}
                    for _, name in ipairs(possible_names) do
                        local dll_path = path.join(vcpkg_bin, name)
                        if os.isfile(dll_path) then
                            os.cp(dll_path, "python/llaisys/libllaisys/")
                            print("Copied OpenBLAS DLL: " .. dll_path)
                            copied = true
                            break
                        end
                    end
                end
            end
        end
        
        if is_plat("linux") then
            os.cp("build/linux/x86_64/release/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()