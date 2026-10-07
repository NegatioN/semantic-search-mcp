GOAL: 
What's the best way to setup running EmbeddingGemma2 https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/to embed my local code by semantic meaning? 
Then how can I setup an mcp server that:
1. Re-indexes changed files in the given path periodically
2. allows us or a harness to query the mcp server (with either ANN , cosine similarity or dot product to find relevant code parts) exposed as a tool somehow.

Ideally I'd like to run the EmbeddingGemma2 model as native as possible (no separate app installs, e.g. ollama, but instead something like llama.cpp as a solution for running locally).

Further goals if implementation is successful:
1. Maybe we should instead embed symbols of the code somehow, instead of only single files?
