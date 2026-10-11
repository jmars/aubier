# P-FORK correctness harness: greedy-decode a prompt and dump the full next-token
# logits at every decode position (binary), for top-token-agreement / KL / perplexity
# comparison between the stock and the residency-split MoE path.
set(TARGET moe-logits-dump)
add_executable(${TARGET} moe_logits_dump.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
