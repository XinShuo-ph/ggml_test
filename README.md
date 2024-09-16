# ggml
The original GGML project can be found [here](https://github.com/ggerganov/ggml).  
The most recent remote fetch is [ggerganov/ggml@fbac47b](https://github.com/ggerganov/ggml/commit/fbac47bf5ea0d2f92bc98bbc79a915ad7477e3e0) at `ggml-upstream` branch.

## Build Project
```bash
git clone --recursive https://github.com/NexaAI/nexa-ggml
pip install -r requirements.txt
mkdir build && cd build # Build the examples
cmake ..
cmake --build . --config Release -j 16 # build project
./bin/mlp /home/ubuntu/nexa-ggml/examples/mlp/model/mlp.gguf # run example
```

## fetch remote
```bash
git remote add ggml-upstream https://github.com/ggerganov/ggml.git
git fetch ggml-upstream
git checkout ggml-upstream
git reset --hard ggml-upstream/master
```