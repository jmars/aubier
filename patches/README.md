# patches — our edits to the vendored trees

`aubier` builds against **copies** of llama.cpp under `src/` (19 of them, one per experiment lane; `src/llama.cpp`
is the one `builds/hip` is built from). Those copies are large and several carry their own `.git`, so they are
**git-ignored**: what is versioned here is the delta.

## `ggml-m4/custom-op-m4.patch`
Our **Mistral-Small-4 custom-op hatch** against a vendored ggml, as a unified diff against the pristine copy
`src/llama-route/ggml` (which is a clean checkout of llama.cpp @ `50e3e3e480c80b9ee0ea067e475602d540dc62c8`):

* `GGML_OP_M4_WKB` — the MLA-absorption gemv (measured: 6.9x standalone, then shipped as the default), **on by default** via a tri-state flag;
* `GGML_OP_M4_KQKV` — fused scores+softmax+contraction, then **tiled with online softmax** and **split-K**
  (L62-L66): correct at 256 … 262 144 keys, 18.8x over the first tiled version;
* `GGML_OP_M4_PHASE` — one op, two phases behind an `op_params` selector: `kvwrite` and `wv_b` (L78).

Every one of them: enum value **appended** (no existing op moved), every `GGML_OP_COUNT`-sized table extended, a
**CPU handler that is FATAL** (the CPU-only arm is the numerics oracle and must not take a fused path), and
**capture-safe** (a plain launch with no host synchronisation, so it stays inside the captured graph).
`patch -p1 < custom-op-m4.patch` from `src/llama.cpp/` reproduces the tree; the six `m4-*` sources are copied
alongside so they are readable without applying anything.
