#!/bin/bash
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda update -n base -c defaults conda -y
conda create -n yoloe python=3.10 -y
conda activate yoloe
git clone https://github.com/apple/ml-mobileclip
cd ml-mobileclip
git clone https://github.com/mlfoundations/open_clip.git
pushd open_clip
git apply ../mobileclip2/open_clip_inference_only.patch
cp -r ../mobileclip2/* ./src/open_clip/

pip install -e .
pip install scikit-learn
popd
cd ..
pip install git+https://github.com/huggingface/pytorch-image-models
pip install -r requirements.txt
pip install jupyterlab
