# InstantStyle 👩‍🎨

InstantStyle is a text-to-image generation application designed to preserve style using two models and an IP Adapter. It supports both FLUX and SDXL modes, each available in a separate tab via a user-friendly Gradio interface.

---

## Features

- **Style-Preserving Generation:** Leverages two different pipelines to generate images that maintain style attributes.
- **Flexible Modes:** Choose between FLUX and SDXL modes according to your preference.
- **Customizable Settings:** Adjust image scale, dimensions, seed, and more using advanced options.
- **IP Adapter Integration:** Uses IP Adapter scaling to manage style and layout features.
- **GPU Acceleration:** Supports GPU rendering for faster image generation.

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for faster generation)
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/InstantStyle.git
   cd InstantStyle
   ```
2. Install dependencies::

   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Hugging Face token:
    For Windows:
   ```bash
   set HF_TOKEN=<your_token>
   ```
    For Linux/Mac:
    ```bash
   export HF_TOKEN=<your_token>
   ```
4. Remove @spaces decorator Used to deploy on HuggingFace ZeroGPU infrastructure

### Running the Application
Launch the Gradio app by running:
```bash
gradio app.py
```
After running the command, open the provided URL in your web browser and start generating style-preserved images.

### Configuration
This app uses Gradio SDK version 4.26.0. For additional configuration details, please refer to the Spaces configuration reference.

### License
This project is licensed under the Apache-2.0 License.

### Short Description
Style-Preserving Text-to-Image Generation