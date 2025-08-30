
# 🚀 My GPU Programming Journey

This repository is a **living log** of my journey into **GPU programming** with **CUDA, Triton, and ONNX**, inspired by the book *Parallel Programming and Optimization (PMPP)*.

The goal is simple:

* Learn GPU computing step by step.
* Document everything I practice and read.
* Share **code, notes, and resources** so others can follow along.

Whether you’re new to GPU programming or brushing up, you’ll find tutorials, experiments, and resources here.

---

## 🛠️ Setup

### 1. Hardware + Drivers

* **GPU**: NVIDIA RTX / GTX or any CUDA-capable GPU
* **Drivers**: Install the latest [NVIDIA GPU drivers](https://www.nvidia.com/Download/index.aspx)

### 2. CUDA Toolkit

```bash
# On Ubuntu WSL
sudo apt update
sudo apt install nvidia-cuda-toolkit
nvcc --version  # verify installation
```

[Official CUDA Toolkit Install Guide](https://developer.nvidia.com/cuda-downloads)

### 3. Python + Triton

```bash
pip install torch triton
```

### 4. VS Code + WSL

* Install [VS Code](https://code.visualstudio.com/)
* Install **Remote - WSL** extension
* Connect to Ubuntu from VS Code (this repo is developed on WSL)

---

## 📖 Daily Log

### 🟢 Day 1: Getting Started with CUDA

* Installed **CUDA Toolkit** and set up VS Code with WSL.
* Learned about **threads, blocks, and grids** in GPU execution.
* Practiced first kernel: *vector addition on GPU*.

📂 Code: [`day1_vector_addition.cu`](day1_vector_addition.cu)
🔗 Resources:

* [NVIDIA CUDA Programming Model Intro](https://developer.nvidia.com/cuda-zone)
* YouTube: [Intro to CUDA Programming](https://youtu.be/3U9M1L8uI4w)

---

### 🟢 Day 2: Memory Hierarchy (Registers, Shared, Global, L1/L2)

* Studied CUDA memory hierarchy.
* Benchmarked performance differences between **global vs shared memory**.
* Wrote kernel for *matrix multiplication* using shared memory.

📂 Code: [`day2_matrix_multiplication.cu`](day2_matrix_multiplication.cu)
🔗 Resources:

* PMPP Chapter 2 (Memory Hierarchy)
* YouTube: [CUDA Memory Hierarchy Explained](https://youtu.be/eR-VQG5QFJg)

---

### 🟢 Day 3: Triton Basics

* Installed **Triton** and ran first kernel.
* Compared Triton vs CUDA in terms of ease of use.
* Implemented *vector add in Triton*.

📂 Code: [`day3_triton_vector_add.py`](day3_triton_vector_add.py)
🔗 Resources:

* [Triton Official Docs](https://triton-lang.org/)
* YouTube: [OpenAI Triton Tutorial](https://youtu.be/9mZpX9yH7cI)

---

### 🟢 Day 4: Optimizing Kernels (Occupancy & Warps)

* Learned about **warps (32 threads)** in CUDA.
* Used `nvprof` to analyze kernel occupancy.
* Started optimizing matrix multiplication.

📂 Code: [`day4_kernel_optimizations.cu`](day4_kernel_optimizations.cu)
🔗 Resources:

* PMPP Chapter 3 (Performance)
* Blog: [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

---

### 🟢 Day 5: ONNX Runtime + GPU Execution

* Exported a PyTorch model to **ONNX**.
* Ran inference using **ONNX Runtime GPU Execution Provider**.
* Benchmarked CPU vs GPU latency.

📂 Code: [`day5_onnx_runtime_gpu.py`](day5_onnx_runtime_gpu.py)
🔗 Resources:

* [ONNX Runtime GPU Docs](https://onnxruntime.ai/)
* YouTube: [ONNX Runtime Explained](https://youtu.be/Ef09PZ9d2C0)

---

## 📚 Learning Resources

### 📖 Books

* **Parallel Programming and Optimization (PMPP)**
* *Programming Massively Parallel Processors* by David Kirk & Wen-mei Hwu

### 🎥 YouTube Channels

* [NVIDIA Developer](https://www.youtube.com/user/NVIDIADeveloper)
* [CoffeeBeforeArch (CUDA tutorials)](https://www.youtube.com/c/CoffeeBeforeArch)
* [OpenAI Triton Talks](https://www.youtube.com/@OpenAITriton)

### 🧑‍💻 Blogs & Docs

* [CUDA Toolkit Docs](https://docs.nvidia.com/cuda/)
* [Triton Docs](https://triton-lang.org/)
* [ONNX Runtime Docs](https://onnxruntime.ai/)

---

Would you like me to make a **starter folder structure + first sample `day1_vector_addition.cu` file** so your repo is immediately clone-and-run ready?
