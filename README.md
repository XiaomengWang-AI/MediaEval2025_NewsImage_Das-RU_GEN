# MediaEval2025_NewsImage_Das-RU_GEN
## Project Overview

This project is developed for the MediaEval 2025 NewsImages challenge, focusing on the image generation subtask.  
(Reference: [MediaEval 2025 NewsImages Task Link](https://multimediaeval.github.io/editions/2025/tasks/newsimages/))

To address the challenge, we chose the Stable Diffusion XL (SDXL) model provided by Hugging Face as the base model.   
(Reference: (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)))  
The core implementation is in the `SDXL.py` file.

To improve facial detail generation and reduce factual errors in the generated images, we experiment with two enhancement strategies:
- Negative prompting – implemented in `SDXLNEG.py`
- Refiner-based Enhancement – implemented in `SDXLREF.py`

These approaches aim to improve both the visual quality and factual alignment of the generated news-related images.
