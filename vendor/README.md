# vendor/ — the ggml sources this engine builds against, pruned to what it uses

`vendor/llama.cpp/` is llama.cpp at **`50e3e3e480c80b9ee0ea067e475602d540dc62c8`** with our three custom ops
applied (see `../patches/ggml-m4/`). Its own `.git` is **not** part of this repository — the tree is plain
tracked files.

**It is pruned to the libraries.** The engine links exactly four of them (`libggml.so`, `libggml-base.so`,
`libggml-cpu.so`, `libggml-hip.so`) and includes only `ggml/include/*.h`. Removed from the upstream tree:
`models/` (36 MB of tokenizer-vocab test fixtures), `tests/`, `examples/`, `tools/` (11 MB of CLI front-ends),
`docs/`, `media/`, `benches/`, `conversion/`, `gguf-py/`, `.github/`, `.devops/`. **3 876 files / 170.6 MB →
1 933 files / 34.8 MB.**

What remains is required by the upstream CMake to configure and build those libraries: `ggml/` (the library),
`src/` (libllama — not linked by the engine, but `add_subdirectory(src)` is unconditional), `common/`,
`vendor/` (submodule stubs `add_subdirectory(vendor)` needs), `scripts/` (generates `build-info.h`), and the
top-level CMake files.

**Because of the prune, build the library targets, not `all`** — `all` also wants the tests, examples, tools and
`app/` sources that are no longer here:

```sh
sh scripts/build_hip.sh          # passes LLAMA_BUILD_TESTS/EXAMPLES/TOOLS/SERVER=OFF and builds
                                 # --target ggml ggml-base ggml-cpu ggml-hip
```

To go back to a full upstream tree, clone it and apply the patch instead:

```sh
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp.full
cd vendor/llama.cpp.full && git checkout 50e3e3e480c80b9ee0ea067e475602d540dc62c8
patch -p1 < ../../patches/ggml-m4/custom-op-m4.patch
```
