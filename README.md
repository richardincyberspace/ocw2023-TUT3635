# Productionizing and Scaling Machine Learning Models on OCI [TUT3635]

In this guide, we will show you how to deploy NVIDIA Triton with TensorRT-LLM backend on Oracle Cloud Infrastructure (OCI).

TensorRT-LLM is currently pre-GA and can be obtained through NVIDIA's Partners Portal.

For this tutorial, you will need two files: 
[tensorrt_llm_backend_aug-release-v1.tar.gz] (https://partners.nvidia.com/DocumentDetails?DocID=1105343)
[tensorrt_llm_aug-release-v1.tar.gz (CUDA 12.1)] (https://partners.nvidia.com/DocumentDetails?DocID=1105342)

Please download both files to `${HOME}/trt-llm/`

### Prerequisites
1. An OCI GPU instance.  We used VM.GPU.A10.2 shape, which has 2 NVIDIA A10 GPUs
2. NVIDIA GPU driver
3. Docker and NVDIA GPU Toolkit
4. Llama 2 13B model in HuggingFace format downloaded to directory: `${HOME}/trt-llm/models/Llama-2-13b-hf`

### Build Triton container with TRT-LLM backend
```bash
cd ${HOME}/trt-llm/
tar xvf tensorrt_llm_backend_aug-release-v1.tar.gz
cd tensorrt_llm_backend_aug-release-v1
tar xvf ../tensorrt_llm_aug-release-v1.tar.gz

docker build -t tensort_llm_backend -f dockerfile/Dockerfile.trt_llm_backend .
```


