#include "llaisys/models/qwen2.h"
#include "../models/qwen2.hpp"

#include <cstdlib>
#include <cstring>

__C {
    // Create model
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {

        if (meta == nullptr) {
            return nullptr;
        }

        // allocate C handle (the C header defines struct LlaisysQwen2Model { void *model; };)
        struct LlaisysQwen2Model *cmodel = (struct LlaisysQwen2Model*)malloc(sizeof(struct LlaisysQwen2Model));
        if (!cmodel) return nullptr;
        cmodel->model = nullptr;

        try {
            // NOTE: Qwen2Model is in namespace llaisys::models and the ctor signature in your C++ impl is:
            // Qwen2Model(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
            auto cpp_model = new llaisys::models::Qwen2Model(meta, device, device_ids, ndevice);
            cmodel->model = reinterpret_cast<void*>(cpp_model);
        } catch (...) {
            free(cmodel);
            return nullptr;
        }
        return cmodel;
    }

    // Destroy model
    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (model == nullptr) return;
        if (model->model) {
            delete reinterpret_cast<llaisys::models::Qwen2Model*>(model->model);
            model->model = nullptr;
        }
        free(model);
    }

    // Infer
    int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        const int64_t *token_ids,
        size_t ntoken,
        int64_t *out_ids,
        size_t out_capacity,
        float temperature,
        int top_k,
        float top_p
    ) {
        if (model == nullptr || model->model == nullptr){
            return -1;

        } 
        auto cpp = reinterpret_cast<llaisys::models::Qwen2Model*>(model->model);
        // The C++ impl uses signature: int64_t infer(const int64_t*, size_t, int64_t*, size_t)
        return cpp->infer(token_ids, ntoken, out_ids, out_capacity, temperature, top_k, top_p);
    }

    // Load a tensor into the model (called from Python loader)
    int llaisysQwen2ModelLoadTensor(
        struct LlaisysQwen2Model *model,
        const char *name,
        const void *data,
        int ndim,
        const int64_t *shape
    ) {
        if (model == nullptr) {
            return -1;
        }
        // guard model->model before dereferencing
        // note: accessing model->model requires model to be valid
        if (model->model == nullptr) {
            return -1;
        }

        try {
            auto cpp = reinterpret_cast<llaisys::models::Qwen2Model*>(model->model);
            return cpp->load_tensor(name, data, ndim, shape);
        } catch (...) {
            return -1;
        }
    }

}