#include "common.cuh"

// MISTRAL-SMALL-4-ONLY (the aubier engine's M1 H3-U5 custom-op hatch).  The fused attention
// core as a single specialized kernel: F16 kcache [320, n_kv_max] x F32 Qcur [320, 32, 1]
// -> scores -> masked softmax -> contraction over the KV extent with V (kcache rows [0,256))
// -> F32 [256, 32, 1].  A plain launch (no host sync, no allocation), so it is legal inside a
// captured CUDA graph.
void ggml_cuda_m4_kqkv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
