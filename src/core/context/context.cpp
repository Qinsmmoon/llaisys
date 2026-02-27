#include "context.hpp"
#include "../../utils.hpp"
#include <thread>
#include <iostream>
#include <sstream>

namespace llaisys::core {

Context::Context() {
    // All device types, put CPU at the end
    std::vector<llaisysDeviceType_t> device_typs;
    for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++) {
        device_typs.push_back(static_cast<llaisysDeviceType_t>(i));
    }
    device_typs.push_back(LLAISYS_DEVICE_CPU);

    // Create runtimes for each device type.
    // Activate the first available device. If no other device is available, activate CPU runtime.
    for (auto device_type : device_typs) {
        const LlaisysRuntimeAPI *api_ = llaisysGetRuntimeAPI(device_type);

        if (api_ == nullptr) {
            std::cerr << "[Context DEBUG] No runtime API registered for device_type=" 
                      << static_cast<int>(device_type) << std::endl;
            // Ensure map has an entry (empty vector)
            _runtime_map[device_type] = std::vector<Runtime *>();
            continue;
        }

        int device_count = 0;
        try {
            device_count = api_->get_device_count();
        } catch (...) {
            std::cerr << "[Context DEBUG] get_device_count() threw for device_type=" 
                      << static_cast<int>(device_type) << std::endl;
            device_count = 0;
        }

        // initialize vector with nullptr entries of size device_count
        std::vector<Runtime *> runtimes_(device_count, nullptr);
        
        for (int device_id = 0; device_id < device_count; device_id++) {
            // 为每个设备 ID 创建运行时
            auto runtime = new Runtime(device_type, device_id);
            runtimes_[device_id] = runtime;
            
            // 如果没有激活的运行时，激活第一个可用的
            if (_current_runtime == nullptr) {
                runtime->_activate();
                _current_runtime = runtime;
            }
        }
        _runtime_map[device_type] = runtimes_;
    }
}

Context::~Context() {
    // Destroy current runtime first.
    delete _current_runtime;

    for (auto &runtime_entry : _runtime_map) {
        std::vector<Runtime *> runtimes = runtime_entry.second;
        for (auto runtime : runtimes) {
            if (runtime != nullptr && runtime != _current_runtime) {
                runtime->_activate();
                delete runtime;
            }
        }
        runtimes.clear();
    }
    _current_runtime = nullptr;
    _runtime_map.clear();
}

void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    // If doest not match the current runtime.
    if (_current_runtime == nullptr || _current_runtime->deviceType() != device_type || _current_runtime->deviceId() != device_id) {
        auto runtimes = _runtime_map[device_type];

        // New clearer checks:
        if (runtimes.empty()) {
            std::ostringstream oss;
            oss << "No devices available for device type " << static_cast<int>(device_type)
                << ". requested device_id=" << device_id;
            throw std::invalid_argument(oss.str());
        }
        if (device_id < 0 || static_cast<size_t>(device_id) >= runtimes.size()) {
            std::ostringstream oss;
            oss << "Invalid device id " << device_id << " for device type " << static_cast<int>(device_type)
                << ". available_count=" << runtimes.size();
            throw std::invalid_argument(oss.str());
        }

        if (_current_runtime != nullptr) {
            _current_runtime->_deactivate();
        }
        if (runtimes[device_id] == nullptr) {
            runtimes[device_id] = new Runtime(device_type, device_id);
        }
        runtimes[device_id]->_activate();
        _current_runtime = runtimes[device_id];
    }
}

Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    return *_current_runtime;
}

// Global API to get thread-local context.
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}

} // namespace llaisys::core