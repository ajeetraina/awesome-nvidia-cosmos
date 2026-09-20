<div align="center">

# 🌌 Awesome NVIDIA Cosmos

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License](https://img.shields.io/badge/License-CC0-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Last Updated](https://img.shields.io/badge/Updated-September%202026-blue.svg)](#)
[![Curated by an Agent](https://img.shields.io/badge/Curated%20by-Cosmos%20Agent%20🤖-76b900.svg)](auto-curator-agent/cosmos-curator.yaml)

A curated list of **NVIDIA Cosmos** - world foundation models (WFMs), datasets, tools, and projects for building **Physical AI**.
Open weights · Open data · Built for robots, autonomous vehicles, and smart infrastructure.

[Models](#-models) • [GitHub Projects](#-github-projects) • [Tools](#-tools--deployment) • [Tutorials](#-tutorials--starter-kits) • [HuggingFace × Cosmos](#-huggingface--cosmos) • [Use Cases](#-real-world-use-cases--companies) • [Videos](#-videos--talks) • [Papers](#-papers) • [Events](#-events) • [Community](#-community)

</div>

## Contents

- [What is NVIDIA Cosmos?](#-what-is-nvidia-cosmos)
- [Model Generations at a Glance](#-model-generations-at-a-glance)
- [Models](#-models)
  - [Cosmos 3 (Latest - Omni-Model)](#cosmos-3-latest---omni-model)
  - [Cosmos Predict (World Generation)](#cosmos-predict-world-generation)
  - [Cosmos Transfer (Controllable / Sim-to-Real)](#cosmos-transfer-controllable--sim-to-real)
  - [Cosmos Reason (Physical Reasoning VLM)](#cosmos-reason-physical-reasoning-vlm)
  - [Cosmos Tokenizer](#cosmos-tokenizer)
  - [Cosmos Guardrails](#cosmos-guardrails)
- [GitHub Projects](#-github-projects)
  - [Official NVIDIA Repositories](#official-nvidia-repositories)
  - [NVIDIA Labs & Blueprints](#nvidia-labs--blueprints)
  - [Community - Robotics & Embodied AI](#community---robotics--embodied-ai)
  - [Community - ComfyUI & Diffusion Tooling](#community---comfyui--diffusion-tooling)
  - [Community - Edge, Deployment & Apps](#community---edge-deployment--apps)
  - [Community - Research, Distillation & Data](#community---research-distillation--data)
- [Tools & Deployment](#-tools--deployment)
- [Tutorials & Starter Kits](#-tutorials--starter-kits)
- [HuggingFace × Cosmos](#-huggingface--cosmos)
- [Real-World Use Cases & Companies](#-real-world-use-cases--companies)
- [Videos & Talks](#-videos--talks)
- [Papers](#-papers)
- [Events](#-events)
- [Community](#-community)
- [Contributing](#-contributing)

---

## 🔭 What is NVIDIA Cosmos?

**NVIDIA Cosmos™** is an open platform of **world foundation models (WFMs)**, datasets, and tools for building **Physical AI** - AI that perceives, reasons about, and acts in the physical world (robotics, autonomous vehicles, and smart infrastructure). Cosmos WFMs generate and predict physically-plausible video "worlds" from text, image, video, and control inputs - primarily to produce controllable synthetic training data and to serve as reasoning and policy backbones for embodied agents. First unveiled at CES on January 6, 2025.

Key properties across the family:

- **Open weights + open source** - checkpoints published on [Hugging Face](https://huggingface.co/nvidia); source code on GitHub. Weights use the permissive **NVIDIA Open Model License** (commercially usable, derivatives allowed).
- **Three functional pillars** - **Predict** (world / video generation and future-state prediction), **Transfer** (controllable conditional generation and sim-to-real), and **Reason** (physical-common-sense reasoning VLM). **Cosmos 3** unifies all three into a single omni-model.
- **Physical AI first** - built to generate synthetic training data, bridge the sim-to-real gap, and provide embodied reasoning for robots and autonomous vehicles.
- **Deployable anywhere** - download from Hugging Face, run as [NVIDIA NIM microservices](https://build.nvidia.com/models?q=cosmos), fine-tune from GitHub, or scale on NVIDIA DGX Cloud.
- **Full ecosystem** - visual tokenizers, guardrails, RL post-training (`cosmos-rl`), distributed data pipelines (`cosmos-xenna`), curation, evaluation, and recipes (`cosmos-cookbook`).

> "The ChatGPT moment for robotics is coming." - Jensen Huang, NVIDIA CEO (CES 2025)

---

## 📊 Model Generations at a Glance

| Model | Gen | Released | Sizes | Role | License |
|---|---|---|---|---|---|
| [Cosmos 3 (Super / Nano / Edge)](#cosmos-3-latest---omni-model) | 3.0 | Jun 2026 | 64B / 16B / 4B | Omni-model: reasoning + world + action (Mixture-of-Transformers) | Open Model |
| [Cosmos-Predict2.5](#cosmos-predict-world-generation) | 2.x | Oct 2025 (14B Dec 2025) | 2B / 14B | Unified Text/Image/Video → World; uses Reason1 as text encoder | Open Model |
| [Cosmos-Transfer2.5](#cosmos-transfer-controllable--sim-to-real) | 2.x | Oct 2025 | 2B | Multi-control world generation; ~3.5× smaller than Transfer1 | Open Model |
| [Cosmos-Reason2](#cosmos-reason-physical-reasoning-vlm) | 2.x | 2025–26 | - | Next-gen physical-reasoning / embodied VLM | Open Model |
| [Cosmos-Predict2](#cosmos-predict-world-generation) | 2.x | Jun 2025 *(archived)* | 0.6B / 2B / 14B | Text2Image + Video2World WFMs | Open Model |
| [Cosmos-Reason1](#cosmos-reason-physical-reasoning-vlm) | 1.x | May 2025 | 7B | Physical-common-sense reasoning VLM (long CoT) | Open Model |
| [Cosmos-Transfer1](#cosmos-transfer-controllable--sim-to-real) | 1.x | 2025 | 7B | World-to-world transfer / sim-to-real (depth, seg, edge, LiDAR, HDMap) | Open Model |
| [Cosmos-Predict1](#cosmos-predict-world-generation) | 1.x | Jan 2025 | 4B–14B | First-gen diffusion + autoregressive WFMs | Open Model |
| [Cosmos-Tokenizer](#cosmos-tokenizer) | 0.1 / 1.0 | Jan 2025 | - | Continuous / discrete image & video tokenizers | Open Model |

---

## 🤖 Models

### Cosmos 3 (Latest - Omni-Model)

Announced June 1, 2026 (GTC Taipei). An **omnimodal world model** built on a unified **Mixture-of-Transformers (MoT)** architecture that jointly models text, image, video, audio, and action - combining an autoregressive transformer for reasoning with a diffusion transformer for generation (unified via 3D RoPE). It acts as a **Reasoner** (grounding, physical reasoning, planning, action forecasting) and a **Generator** (future/world prediction, synthetic data, policy learning).

| Model | Params | Description | Links |
|---|---|---|---|
| **Cosmos3-Super** | 64B | Data-center scale; highest physics accuracy; teacher / synthetic-data model | [HF](https://huggingface.co/collections/nvidia/cosmos3) · [GitHub](https://github.com/NVIDIA/cosmos) |
| **Cosmos3-Nano** | 16B | Balanced speed/quality; sub-second reasoning | [HF](https://huggingface.co/collections/nvidia/cosmos3) |
| **Cosmos3-Edge** | 4B | Real-time edge robotics | [HF](https://huggingface.co/collections/nvidia/cosmos3) |

- Paper: [Cosmos 3: Omnimodal World Models for Physical AI (arXiv:2606.02800)](https://arxiv.org/abs/2606.02800)
- Repo: [github.com/NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) · NIM: [build.nvidia.com](https://build.nvidia.com/models?q=cosmos) · Blog: [Develop Physical AI Reasoning, World, and Action Models with Cosmos 3](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/)

---

### Cosmos Predict (World Generation)

General-purpose WFMs that generate and predict future world states as video, from text / image / video inputs. Fine-tunable into customized downstream world models.

| Model | Released | Sizes | Description | Links |
|---|---|---|---|---|
| **Cosmos-Predict2.5** | Oct 2025 (14B Dec 2025) | 2B / 14B | Flow-based model unifying Text2World, Image2World, and Video2World; uses Cosmos-Reason1 as its text encoder; 720p @ 16fps | [HF 2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) · [HF 14B](https://huggingface.co/nvidia/Cosmos-Predict2.5-14B) · [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5) |
| **Cosmos-Predict2** *(archived)* | Jun 2025 | 0.6B / 2B / 14B | Improved WFMs; Text2Image + Video2World; native ComfyUI support | [HF](https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959) · [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2) |
| **Cosmos-Predict1** | Jan 2025 | 4B–14B | First-gen diffusion (Text2World, Video2World, WorldInterpolator) + autoregressive WFMs | [GitHub](https://github.com/nvidia-cosmos/cosmos-predict1) |

---

### Cosmos Transfer (Controllable / Sim-to-Real)

World-to-world transfer models that generate photorealistic worlds conditioned on structured control inputs - bridging the perceptual gap between simulation and reality for data augmentation.

| Model | Released | Size | Description | Links |
|---|---|---|---|---|
| **Cosmos-Transfer2.5** | Oct 2025 | 2B | Built on Predict2.5; multi-spatial control (RGB, depth, segmentation, edge, blur via JSON control specs); ~3.5× smaller than Transfer1 with higher fidelity | [HF](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) · [GitHub](https://github.com/nvidia-cosmos/cosmos-transfer2.5) · [NIM](https://build.nvidia.com/nvidia/cosmos-transfer2_5-2b/modelcard) |
| **Cosmos-Transfer1** | 2025 | 7B | World-to-world transfer; control via segmentation, depth, canny edge, blur, LiDAR, HDMap, keypoint; 4K upscaler + single-step distilled variant | [HF](https://huggingface.co/nvidia/Cosmos-Transfer1-7B) · [GitHub](https://github.com/nvidia-cosmos/cosmos-transfer1) |

---

### Cosmos Reason (Physical Reasoning VLM)

Vision-language models that understand physical common sense and generate embodied decisions through long chain-of-thought reasoning. Also used as the text encoder for Cosmos-Predict2.5.

| Model | Released | Size | Description | Links |
|---|---|---|---|---|
| **Cosmos-Reason2** | 2025–26 | - | Next-gen physical-common-sense embodied reasoning models | [GitHub](https://github.com/nvidia-cosmos/cosmos-reason2) |
| **Cosmos-Reason1** | May 2025 | 7B | Reasoning VLM for spatial-temporal / embodied reasoning; trained via vision pre-training → SFT → Physical AI RL | [HF](https://huggingface.co/nvidia/Cosmos-Reason1-7B) · [GitHub](https://github.com/nvidia-cosmos/cosmos-reason1) · [Collection](https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7) |

---

### Cosmos Tokenizer

A suite of image and video neural tokenizers used across the Cosmos WFMs.

| Type | Description | Links |
|---|---|---|
| **Continuous (CI / CV)** | Latent embeddings for diffusion models | [GitHub](https://github.com/NVIDIA/Cosmos-Tokenizer) · [Research](https://research.nvidia.com/labs/cosmos-lab/cosmos-tokenizer/) |
| **Discrete (DI / DV)** | Quantized codes (FSQ) for autoregressive models | e.g. [`nvidia/Cosmos-1.0-Tokenizer-DV8x16x16`](https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-DV8x16x16) |

- Spatial compression 8× / 16×, temporal 4× / 8× (up to ~2048× total); Haar-wavelet encoder front-end. Documented in the [Cosmos WFM Platform paper](https://arxiv.org/abs/2501.03575).

---

### Cosmos Guardrails

Pre- and post-generation safety models (including a RetinaFace-based face-blur filter), enabled by default in the Cosmos pipelines.

| Model | Description | Links |
|---|---|---|
| **Cosmos-Guardrail1** | Safety guardrails for the 2.x generation pipelines | [HF](https://huggingface.co/nvidia/Cosmos-Guardrail1) |
| **Cosmos-1.0-Guardrail** | Safety guardrails for the 1.x generation pipelines | [HF](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) |

---

## 🛠 GitHub Projects

The core ask of this list - real, meaningful projects built with or on top of NVIDIA Cosmos. Star counts are approximate. Both `NVIDIA/*` (Cosmos 3 era) and `nvidia-cosmos/*` (Predict/Transfer/Reason 1–2.5) orgs are official.

### Official NVIDIA Repositories

| Repo | Stars | Description |
|---|---|---|
| [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) | ~11.7k | Main Cosmos 3 platform: omnimodal world models, datasets, inference/training/eval + cookbooks - the current flagship repo |
| [NVIDIA/Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) | ~1.7k | Suite of image/video neural tokenizers *(archived; folded into NVIDIA/cosmos)* |
| [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | ~1.4k | Latest Predict WFM - flow-based video future-state prediction; includes AV-post-trained checkpoints |
| [nvidia-cosmos/cosmos-reason1](https://github.com/nvidia-cosmos/cosmos-reason1) | ~960 | 7B physical-reasoning VLM (Qwen2.5-VL based) for embodied decisions via chain-of-thought |
| [nvidia-cosmos/cosmos-transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1) | ~820 | World-to-world transfer (sim2real) conditioned on segmentation/depth/edge; AV LiDAR/HDMap variant |
| [nvidia-cosmos/cosmos-predict2](https://github.com/nvidia-cosmos/cosmos-predict2) | ~790 | Predict2 general-purpose WFMs *(archived; ComfyUI-supported)* |
| [nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | ~730 | Transfer2.5 built on Predict2.5 - multi-input spatial-control world simulation |
| [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework) | ~505 | Inference and training framework to run the Cosmos models |
| [nvidia-cosmos/cosmos-cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook) | ~470 | Post-training scripts, recipes, and samples for the Cosmos ecosystem |
| [nvidia-cosmos/cosmos-predict1](https://github.com/nvidia-cosmos/cosmos-predict1) | ~470 | First-gen general-purpose WFMs, fine-tunable into custom world models |
| [nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) | ~470 | Async RL/SFT + RLHF post-training framework specialized for Physical AI (used for Cosmos-Reason) |
| [nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2) | ~440 | Next-gen physical-common-sense embodied reasoning models |
| [NVIDIA/cosmos-curator](https://github.com/NVIDIA/cosmos-curator) | ~265 | Distributed video curation system that powers Cosmos training-data generation |
| [nvidia-cosmos/cosmos-xenna](https://github.com/nvidia-cosmos/cosmos-xenna) | ~85 | Python library for distributed data pipelines on Ray (Cosmos data infra) |
| [NVIDIA/cosmos-evaluator](https://github.com/NVIDIA/cosmos-evaluator) | ~47 | Automated evaluation & grading of synthetic video output from Cosmos models |

### NVIDIA Labs & Blueprints

| Repo | Stars | Description |
|---|---|---|
| [NVIDIA-AI-Blueprints/video-search-and-summarization](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) | ~1.8k | GPU video-analytics agent blueprint using Cosmos VLMs + Nemotron + RAG/NIMs |
| [nv-tlabs/omni-dreams](https://github.com/nv-tlabs/omni-dreams) | ~330 | Cosmos-Dreams - real-time photorealistic video world model for AV simulation (Toronto AI Lab) |
| [NVIDIA-Omniverse-blueprints/cosmos-dataset-search](https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search) | ~95 | Semantic search across video datasets for Cosmos data curation |
| [nv-tlabs/cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits) | ~48 | Toolkits for Cosmos-Transfer1-7B-Sample-AV (autonomous-vehicle data) |

### Community - Robotics & Embodied AI

| Repo | Stars | Description |
|---|---|---|
| [doosan-robotics/explainable-palletizer](https://github.com/doosan-robotics/explainable-palletizer) | ~21 | Mixed palletizing with explainable visual reasoning on Cosmos |
| [strands-labs/strands-for-cosmos](https://github.com/strands-labs/strands-for-cosmos) | ~9 | Agent framework giving agents physics understanding + video/audio/action generation via Cosmos |
| [ganatrask/NOVA](https://github.com/ganatrask/NOVA) | ~7 | Voice→Reason→Act pipeline: Parakeet ASR + Cosmos-Reason2 + GR00T on a Reachy 2 humanoid |
| [cagataycali/strands-cosmos](https://github.com/cagataycali/strands-cosmos) | ~5 | Strands agent integration giving agents physics-aware perception via Cosmos |
| [cagataycali/thor-cosmos](https://github.com/cagataycali/thor-cosmos) | ~1 | Strands agent orchestrating the Cosmos ecosystem on Jetson AGX Thor for real-time robot perception |
| [doosan-robotics/palletizing-ai](https://github.com/doosan-robotics/palletizing-ai) | ~1 | AI-powered optimal palletizing using Cosmos-Reason2 physical reasoning |
| [naveentnj/cosmos-embodied-ai](https://github.com/naveentnj/cosmos-embodied-ai) | - | Physical AI sim/eval framework for Cosmos edge world-action models with Isaac Sim/Lab |

### Community - ComfyUI & Diffusion Tooling

| Repo | Stars | Description |
|---|---|---|
| [Mirumo0u0/ComfyUI-Cosmos-Reference](https://github.com/Mirumo0u0/ComfyUI-Cosmos-Reference) | ~54 | Adds image-reference feature to Cosmos / "Anima" models in ComfyUI |
| [KeithZ117/Comfyui-anima-sampler](https://github.com/KeithZ117/Comfyui-anima-sampler) | ~54 | Cosmos-style RF sampler (FlowUniPC + PC3) for Anima in ComfyUI |
| [RyukoMatoiFan/ComfyUI-Cosmos3](https://github.com/RyukoMatoiFan/ComfyUI-Cosmos3) | ~13 | Custom nodes for Cosmos3-Nano (text/image-to-video + audio) |
| [Anzhc/Anzhc-ComfyUI-Cosmos-Reference](https://github.com/Anzhc/Anzhc-ComfyUI-Cosmos-Reference) | ~8 | Cosmos reference-image nodes for ComfyUI |
| [NicholaiVogel/comfyui-materia](https://github.com/NicholaiVogel/comfyui-materia) | ~6 | Diffusion inverse rendering (RGB→PBR maps) via NVIDIA Cosmos 7B |
| [rikunarita/ComfyUI-ModelMergeCosmosPredict2-2B-Slerp](https://github.com/rikunarita/ComfyUI-ModelMergeCosmosPredict2-2B-Slerp) | ~1 | SLERP model-merge node for Cosmos-Predict2-2B |

### Community - Edge, Deployment & Apps

| Repo | Stars | Description |
|---|---|---|
| [kabilankb/cosmos3-nano-gb10](https://github.com/kabilankb/cosmos3-nano-gb10) | ~5 | Runs Cosmos3-Nano (16B) on Dell Pro Max GB10 (ARM Blackwell) for text/image-to-video |
| [tuttlebr/cosmos-gradio-app](https://github.com/tuttlebr/cosmos-gradio-app) | - | Gradio web app for physics-aware video generation with Cosmos WFMs |
| [chengchencon/Cosmos-UserGuide](https://github.com/chengchencon/Cosmos-UserGuide) | ~6 | Install/use guide for the NVIDIA Cosmos platform |
| [eivholt/edgeai-synthetic-cosmos-predict](https://github.com/eivholt/edgeai-synthetic-cosmos-predict) | ~6 | Walkthrough: edge-AI object detection trained on Cosmos-Predict2 synthetic images |

### Community - Research, Distillation & Data

| Repo | Stars | Description |
|---|---|---|
| [csy2077/data-forcing-distillation](https://github.com/csy2077/data-forcing-distillation) | ~56 | Few-step video-gen distillation; image-to-video built on Cosmos |
| [andreaskoepf/cosmos3-dk1](https://github.com/andreaskoepf/cosmos3-dk1) | ~5 | Training config for the Cosmos 3 model |
| [lowweihong/cosmos-data-analyzer](https://github.com/lowweihong/cosmos-data-analyzer) | - | Agentic pipeline diagnosing model failures + curating training data for Cosmos WFMs |
| [StaryMoon/Cosmos-WFM-Unofficial](https://github.com/StaryMoon/Cosmos-WFM-Unofficial) | - | Unofficial PyTorch reproduction of the Cosmos WFM platform |

> **Ecosystem repos that frequently integrate Cosmos:** [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (robot foundation models), [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) (robot learning in simulation), [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator) (data curation).

---

## 🚀 Tools & Deployment

| Tool | Description | Link |
|---|---|---|
| **NVIDIA NIM (Cosmos)** | Optimized inference microservices for Cosmos models | [build.nvidia.com](https://build.nvidia.com/models?q=cosmos) |
| **cosmos-rl** | Async RL / SFT / RLHF post-training framework for Physical AI | [GitHub](https://github.com/nvidia-cosmos/cosmos-rl) |
| **cosmos-xenna** | Distributed data pipelines on Ray for large-scale video curation | [GitHub](https://github.com/nvidia-cosmos/cosmos-xenna) |
| **cosmos-curator** | Video curation system that processes, analyzes, and organizes training video | [GitHub](https://github.com/NVIDIA/cosmos-curator) |
| **cosmos-evaluator** | Automated evaluation & grading of synthetic Cosmos video output | [GitHub](https://github.com/NVIDIA/cosmos-evaluator) |
| **cosmos-cookbook** | Post-training scripts, recipes, and samples | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook) |
| **ComfyUI (Cosmos-Predict2)** | Native ComfyUI support for Cosmos-Predict2 video generation | [Examples](https://comfyanonymous.github.io/ComfyUI_examples/cosmos_predict2/) |
| **Isaac Sim + Cosmos (Replicator)** | Generate synthetic data from 3D scenes with Cosmos Transfer | [Tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_cosmos.html) |

---

## 📚 Tutorials & Starter Kits

| Title | Source | Description |
|---|---|---|
| [NVIDIA Cosmos for Developers](https://developer.nvidia.com/cosmos) | developer.nvidia.com | Main developer landing page - models, NIMs, and docs |
| [Cosmos Documentation](https://docs.nvidia.com/cosmos/latest/) | docs.nvidia.com | Official docs covering Predict, Transfer, and Reason |
| [Develop Custom Physical AI Models with Cosmos Predict-2](https://developer.nvidia.com/blog/develop-custom-physical-ai-foundation-models-with-nvidia-cosmos-predict-2/) | NVIDIA Blog | Building / post-training custom world models with Predict-2 |
| [Curating Synthetic Datasets with Cosmos Reason](https://developer.nvidia.com/blog/curating-synthetic-datasets-to-train-physical-ai-models-with-nvidia-cosmos-reason/) | NVIDIA Blog | Using Cosmos Reason as a VLM to curate/filter synthetic data |
| [Simplify AV Development with New Cosmos WFMs](https://developer.nvidia.com/blog/simplify-end-to-end-autonomous-vehicle-development-with-new-nvidia-cosmos-world-foundation-models/) | NVIDIA Blog | End-to-end autonomous-vehicle workflow with Cosmos |
| [Scale Data Generation with the Cosmos Cookbook](https://developer.nvidia.com/blog/how-to-scale-data-generation-for-physical-ai-with-the-nvidia-cosmos-cookbook/) | NVIDIA Blog | Recipes for large-scale synthetic data generation |
| [Develop Reasoning, World & Action Models with Cosmos 3](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/) | NVIDIA Blog | Intro and workflows for the Cosmos 3 omnimodal models |
| [Welcome NVIDIA Cosmos 3](https://huggingface.co/blog/nvidia/cosmos-3-for-physical-ai) | Hugging Face | Community walkthrough of Cosmos 3 as the first open omni-model for Physical AI |
| [Isaac Sim, Omniverse & Cosmos Ecosystem Explained](https://www.ridgerun.ai/post/nvidia-isaac-sim-omniverse-and-cosmos-the-robotics-ai-simulation-ecosystem-explained) | RidgeRun | How Isaac Sim, Isaac Lab, Omniverse, and Cosmos fit together |
| [NVIDIA Launches Cosmos 3: Open Frontier Foundation Model for Physical AI](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai) | NVIDIA Newsroom | Official launch announcement for Cosmos 3, the open omnimodal world foundation model unifying reasoning, world generation, and action |

---

## 🤗 HuggingFace × Cosmos

- **Organization:** [huggingface.co/nvidia](https://huggingface.co/nvidia) - all official Cosmos weights
- **Cosmos 3 collection:** [huggingface.co/collections/nvidia/cosmos3](https://huggingface.co/collections/nvidia/cosmos3)
- **Predict2.5:** [2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) · [14B](https://huggingface.co/nvidia/Cosmos-Predict2.5-14B)
- **Predict2 collection:** [link](https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959)
- **Transfer:** [Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) · [Transfer1-7B](https://huggingface.co/nvidia/Cosmos-Transfer1-7B)
- **Reason1:** [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) · [collection](https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7)
- **Guardrails:** [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)
- **ComfyUI repackaged weights:** [Comfy-Org/Cosmos_Predict2_repackaged](https://huggingface.co/Comfy-Org/Cosmos_Predict2_repackaged)

---

## 🏢 Real-World Use Cases & Companies

Early adopters announced at CES 2025 ([NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development)):

| Company | Domain | Use Case | Reference |
|---|---|---|---|
| Uber | Autonomous mobility | Scaling AV data and development with Cosmos | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |
| Waabi | Autonomous vehicles | AV data curation and development | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |
| Wayve | Autonomous driving | Edge-case scenario generation | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |
| Foretellix | AV testing / simulation | Cosmos Transfer + Omniverse for realistic scenario variation | [Foretellix](https://www.foretellix.com/data-automation-toolchain-for-ai-powered-av-development/) |
| 1X | Humanoid robots | World Model Challenge dataset built with Cosmos | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |
| Agility Robotics | Humanoid robots | Robot learning and data generation | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |
| Figure AI | Humanoid robots | Physical AI model development | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |
| Skild AI | General-purpose robots | Robot foundation models | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |
| XPENG | Humanoid robots | Accelerating humanoid development | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |
| Neura Robotics | Cognitive robots | Robot development | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |
| Virtual Incision | Surgical robotics | Robot development | [Newsroom](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-world-foundation-model-platform-to-accelerate-physical-ai-development) |

See also: [NVIDIA Expands Automotive Ecosystem With Physical AI](https://blogs.nvidia.com/blog/auto-ecosystem-physical-ai/) · [Into the Omniverse: WFMs Advance AV Simulation & Safety](https://blogs.nvidia.com/blog/wfm-advance-av-sim-safety/).

---

## 🎥 Videos & Talks

- [CES 2025 Keynote - Jensen Huang unveils NVIDIA Cosmos](https://blogs.nvidia.com/blog/ces-2025-jensen-huang/) - NVIDIA / Jensen Huang (2025)
- [An Introduction to NVIDIA Cosmos World Foundation Models | GTC 2025](https://www.youtube.com/watch?v=kChwwFb5gMU) - Ming-Yu Liu, VP Generative AI Research (2025)
- [An Introduction to NVIDIA Cosmos World Foundation Models (S72431)](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72431/) - NVIDIA On-Demand, GTC (2025)
- [Developing Next-Gen AVs with Physical AI-Powered WFMs (DD40001)](https://www.nvidia.com/en-us/on-demand/session/gtc25-dd40001/) - NVIDIA On-Demand, GTC (2025)
- [Cosmos WFMs for Autonomous Driving Development (S73198)](https://www.nvidia.com/en-us/on-demand/session/gtc25-s73198/) - NVIDIA On-Demand, GTC (2025)
- [Generate Synthetic Data for Physical AI with Cosmos WFMs (DLIT81644)](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81644/) - NVIDIA On-Demand, GTC 2026

---

## 📄 Papers

| Paper | arXiv | Year | Description |
|---|---|---|---|
| **Cosmos World Foundation Model Platform for Physical AI** | [2501.03575](https://arxiv.org/abs/2501.03575) | 2025 | Foundational platform paper: video curation pipeline, pre-trained WFMs, post-training, and video tokenizers |
| **Cosmos-Reason1: From Physical Common Sense to Embodied Reasoning** | [2503.15558](https://arxiv.org/abs/2503.15558) | 2025 | Vision-language models for physical reasoning via long chain-of-thought; 4-stage training |
| **Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control** | [2503.14492](https://arxiv.org/abs/2503.14492) | 2025 | Diffusion world-to-world transfer with multimodal spatial control and adaptive weighting |
| **World Simulation with Video Foundation Models for Physical AI** (Predict2.5) | [2511.00062](https://arxiv.org/abs/2511.00062) | 2025 | Flow-based model unifying Text/Image/Video2World; leverages Cosmos-Reason1 for grounding |
| **Cosmos 3: Omnimodal World Models for Physical AI** | [2606.02800](https://arxiv.org/abs/2606.02800) | 2026 | Omnimodal world models jointly modeling text, image, video, audio, and action in a unified MoT |

Research lab pages: [Cosmos-Predict2](https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2/) · [Cosmos-Tokenizer](https://research.nvidia.com/labs/cosmos-lab/cosmos-tokenizer/) · [Cosmos-Transfer2.5](https://research.nvidia.com/labs/dir/cosmos-transfer2.5/)

---

## 📅 Events

- **NVIDIA GTC** - annual talks and DLI sessions on Cosmos and Physical AI ([on-demand catalog](https://www.nvidia.com/en-us/on-demand/))
- **NVIDIA Cosmos Cookoff** - community hackathon for building on Cosmos ([luma.com/nvidia-cosmos-cookoff](https://luma.com/nvidia-cosmos-cookoff))
- **CES** - where Cosmos was first announced (Jan 2025)

---

## 💬 Community

- **GitHub - Cosmos 3 / flagship:** [github.com/NVIDIA/cosmos](https://github.com/NVIDIA/cosmos)
- **GitHub - WFM org:** [github.com/nvidia-cosmos](https://github.com/nvidia-cosmos)
- **NVIDIA Developer Forums:** [forums.developer.nvidia.com](https://forums.developer.nvidia.com/)
- **Hugging Face:** [huggingface.co/nvidia](https://huggingface.co/nvidia)
- **Product page:** [nvidia.com/en-us/ai/cosmos](https://www.nvidia.com/en-us/ai/cosmos/)
- **Docs:** [docs.nvidia.com/cosmos](https://docs.nvidia.com/cosmos/latest/)

---

## 🤝 Contributing

Contributions are welcome! This list is jointly maintained by the community and an [autonomous curator agent](auto-curator-agent/cosmos-curator.yaml). See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Found a broken link, a missing project, or a new Cosmos release? [Open an issue](../../issues) or submit a PR. The agent also raises PRs automatically - look for the [`agent-bot`](../../pulls?q=label%3Aagent-bot) label.

---

<div align="center">

**License:** [CC0 1.0 Universal](LICENSE) - dedicated to the public domain.

Made with 🌌 for the Physical AI community.

</div>
