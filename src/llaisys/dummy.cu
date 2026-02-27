// dummy.cu
// 声明一个空内核，强制设备链接器包含设备代码
__global__ void dummy_kernel() {}

// 可选：添加一个主机函数调用该内核，确保内核不被丢弃
extern "C" void dummy_force_device_link() {
    dummy_kernel<<<1, 1>>>();
}