# How to use GGML
1. Model Loading: sets up the appropriate backend (CUDA, Metal, or CPU) and create GGML context.
2. Graph Building: creates a computation graph for  the forward pass.
3. Computation:  builds the graph, allocates memory for intermediate tensors. Executes the computation using the specified backend.
4. Memory Management: properly frees allocated resources at the end.

# A list of GGML examples