#include "common.cuh"

// MISTRAL-SMALL-4-ONLY (the aubier engine's M1 H3-U4 custom-op hatch).  The MLA absorption
// gemv as a single specialized kernel: Q8_0 [64, 256, 32] x F32 [64, 32, 1] -> F32 [256, 32, 1].
// A plain launch (no host sync, no allocation), so it is legal inside a captured CUDA graph.
void ggml_cuda_m4_wkb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
