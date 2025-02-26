<div align="center" style="display: block;margin-left: auto;margin-right: auto;width: 70%;">
<h1>InstantStyle FLUX & SDXL</h1>

<h4 align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0.html">
    <img src="https://img.shields.io/badge/license-Apache2.0-blue" alt="savorswipe is under the Apache 2.0 liscense" />
  </a>
  <a href="https://huggingface.co/docs/diffusers/v0.9.0/en/index">
    <img src="https://img.shields.io/badge/HuggingFace%20Diffusers-FFD21E" alt="Google Custom Search" />
  </a>
  <a href="https://www.gradio.app/">
    <img src="https://img.shields.io/badge/Gradio-5+-F97700" alt="Gradio Version" />
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python%203.12+-FAD641" alt="Python Version">
    </a>
</h4>
<p align="center">
  <p align="center"><b>Preserve Style Across text-to-image Generation<br> <a href="https://hatman-instantstyle-flux-sdxl.hf.space"> InstantStyle » </a> </b> </p>
</p>

</div>

## 🌟 Features

- **Dual Model Support**
  - FLUX.1: Optimized for speed and efficiency
  - SDXL: Enhanced quality with InstantStyle integration
- **Style Preservation** using IP-Adapter technology
- **Intuitive Interface** powered by Gradio
- **Advanced Controls** for fine-tuned generation

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/instant-style
cd instant-style
pip install -r requirements.txt
python app.py
```

## 📋 Requirements

```txt
accelerate>=0.20.0
diffusers>=0.21.4
torch>=2.0.0
transformers>=4.30.0
xformers>=0.0.20
sentencepiece
safetensors
gradio>=5.0.0
```

## 🎨 Usage

1. Choose your preferred model (FLUX or SDXL)
2. Upload a reference style image
3. Enter your creative prompt
4. Adjust generation parameters:
   - Image Scale (0-1)
   - Dimensions
   - Guidance Scale
   - Steps
5. Generate your styled image