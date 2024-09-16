# ggml
The original GGML project can be found [here](https://github.com/ggerganov/ggml).

## Build Project
```bash
git clone --recursive https://github.com/NexaAI/nexa-ggml
pip install -r requirements.txt
mkdir build && cd build # Build the examples
cmake ..
cmake --build . --config Release -j 8
```

## fetch remote
```bash
git remote add ggml-upstream https://github.com/ggerganov/ggml.git
git fetch ggml-upstream
git checkout ggml-upstream
git reset --hard ggml-upstream/master
```