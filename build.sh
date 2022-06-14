#!/bin/bash


pushd layernorm; make; popd
cp ./layernorm/LayerNormPlugin.so /target/LayerNormPlugin.so

pushd swish; make; popd
cp ./swish/SwishPlugin.so /target/SwishPlugin.so

pushd glu; make; popd
cp ./glu/GluPlugin.so /target/GluPlugin.so

python -m onnxsim /workspace/encoder.onnx encoder_sim.onnx --input-shape "speech:64,256,80" "speech_lengths:64"  --dynamic-input-shape

python polish.py

python encoder2trt.py

python convert_decoder.py

python decoder2trt.py
