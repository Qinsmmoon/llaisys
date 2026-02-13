from ctypes import (
    Structure,
    POINTER,
    c_size_t,
    c_float,
    c_int64,
    c_int,
    c_char_p,
    c_void_p,
)
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),       # model.embed_tokens.weight
        ("out_embed", llaisysTensor_t),      # lm_head.weight
        ("out_norm_w", llaisysTensor_t),     # model.norm.weight
        
        # 28层，每层12个权重张量
        ("attn_norm_w", POINTER(llaisysTensor_t)),  # [28] input_layernorm.weight
        ("mlp_norm_w", POINTER(llaisysTensor_t)),   # [28] post_attention_layernorm.weight
        ("attn_q_w", POINTER(llaisysTensor_t)),     # [28] self_attn.q_proj.weight
        ("attn_q_b", POINTER(llaisysTensor_t)),     # [28] self_attn.q_proj.bias
        ("attn_k_w", POINTER(llaisysTensor_t)),     # [28] self_attn.k_proj.weight
        ("attn_k_b", POINTER(llaisysTensor_t)),     # [28] self_attn.k_proj.bias
        ("attn_v_w", POINTER(llaisysTensor_t)),     # [28] self_attn.v_proj.weight
        ("attn_v_b", POINTER(llaisysTensor_t)),     # [28] self_attn.v_proj.bias
        ("attn_o_w", POINTER(llaisysTensor_t)),     # [28] self_attn.o_proj.weight
        ("mlp_gate_w", POINTER(llaisysTensor_t)),   # [28] mlp.gate_proj.weight
        ("mlp_up_w", POINTER(llaisysTensor_t)),     # [28] mlp.up_proj.weight
        ("mlp_down_w", POINTER(llaisysTensor_t)),   # [28] mlp.down_proj.weight
    ]


def load_qwen2(lib):
    """
    Bind qwen2 related functions on the provided shared lib (ctypes.CDLL)
    After calling this, the LIB_LLAISYS will have the qwen2 symbols with proper arg/return types.
    """
    # Create: (const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
    lib.llaisysQwen2ModelCreate.argtypes = [POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, POINTER(c_int), c_int]
    lib.llaisysQwen2ModelCreate.restype = c_void_p

    # Destroy
    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    # Load tensor: (model, name, data_ptr, shape_ptr, ndim)
    lib.llaisysQwen2ModelLoadTensor.argtypes = [c_void_p, c_char_p, c_void_p, c_int, POINTER(c_int64)]
    lib.llaisysQwen2ModelLoadTensor.restype = c_int

    # Infer: (model, int64_t* token_ids, size_t ntoken) -> int64_t next token
    lib.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64
    
    return lib