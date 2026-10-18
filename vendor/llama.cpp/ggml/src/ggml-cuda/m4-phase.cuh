#include "common.cuh"

// MISTRAL-SMALL-4-ONLY (the aubier engine's M1 H3-U11 custom-op hatch).  ONE op,
// GGML_OP_M4_PHASE, carrying the attention chain's remaining per-token phase fusions
// selected by op_params[0].  Each phase is a fixed shape, fixed geometry, plain launch
// (no host sync, no allocation), so it is legal inside a captured CUDA graph.
//
//   0 = M4_PHASE_KVWRITE : wkv_a_mqa gemv + attn_kv_a_norm + k_pe rope + Kcur concat +
//                          the set_rows scatter into the KV cache -> ONE node, two launches.
//   1 = M4_PHASE_WVB     : attn_v_b (the MLA value absorption gemv) -> one launch.
//
// The CPU handler for the op is a FATAL (ggml-cpu.c): the CPU arm is the numerics oracle
// and must keep the reference op sequence.
void ggml_cuda_m4_phase(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
