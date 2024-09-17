# ggml
The original GGML project can be found [here](https://github.com/ggerganov/ggml).  
The most recent remote fetch is [ggerganov/ggml@fbac47b](https://github.com/ggerganov/ggml/commit/fbac47bf5ea0d2f92bc98bbc79a915ad7477e3e0) at `ggml-upstream` branch.

## Build Project and run example with CPU backend
```bash
git clone --recursive https://github.com/NexaAI/nexa-ggml
pip install -r requirements.txt
rm -rf build && mkdir build && cd build # Build the examples
cmake ..
cmake --build . --config Release -j16 # build project
```
Run CPU example
```bash
python model.py
python convert.py
./bin/mlp_ctx /home/ubuntu/nexa-ggml/examples/mlp/model/mlp.gguf # run CPU example
```

## Build Project and run example with GPU backend
Run GPU example
```bash
cmake -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
./bin/mlp_backend /home/ubuntu/nexa-ggml/examples/mlp/model/mlp.gguf --backend=cuda # run CUDA example

```
## fetch remote
```bash
git remote add ggml-upstream https://github.com/ggerganov/ggml.git
git fetch ggml-upstream
git checkout ggml-upstream
git reset --hard ggml-upstream/master
```