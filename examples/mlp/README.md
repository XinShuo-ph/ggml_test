# GGML for MLP
Create model and convert it to GGML format.
```
python model.py
python convert.py
```

Build and run the example.
```
cd ~ # Go to root of project
rm -rf build && mkdir build && cd build # Build the examples
cmake ..
cmake --build . --config Release -j16 # build project
./bin/mlp /home/ubuntu/nexa-ggml/examples/mlp/model/mlp.gguf
```
