FROM nvcr.io/nvidia/pytorch:21.03-py3

COPY requirements.txt requirements.txt

# Install additional requirements on top of nvidia container baseline
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0 7.5+PTX 8.0+PTX" pip install --no-cache-dir -r requirements.txt
