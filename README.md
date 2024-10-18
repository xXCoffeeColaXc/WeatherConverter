# Diffusion based Image Translation with Semantic mask Guidance

This repository is the refactored version of [DiffusionBasedImageTranlationWithGradientGuidance](https://github.com/xXCoffeeColaXc/DiffusionBasedImageTranlationWithGradientGuidance/tree/main) !

## Project Description
This repository hosts an implementation of the "Diffusion-based Image Translation with Label Guidance for Domain Adaptive Semantic Segagation" paper.
The project focuses generating urban scene images in different weather conditions, such as rainy, foggy, night from ideal sunny-day images. For semantic consistency we inject semantic mask as gradient guidance to achive this consistency between the input and target image.

## Paper Reference

This implementation is based on the [Diffusion-based Image Translation with Label Guidance for Domain Adaptive Semantic Segmentation](https://arxiv.org/pdf/2308.12350) paper. The segmentation model is a fine-tuned version of the [DeepLabV3Plus repository](https://github.com/VainF/DeepLabV3Plus-Pytorch). All credits to the authors of these works.

## Install

(Currently I'm working on the segmentation model only!)

### Segmentation Model
1. To create a virtual environment in the root directory, use the following command (or do it with conda):
```bash
python3 -m venv venv
```

2. Activate the virtual environment using:
```bash
source venv/bin/activate
```
3. Install the necessary dependencies by running:
```bash
pip install -r /requirements.txt
```

## Notes
- The PTL algorithm described in the paper is yet to be implemented.

---

If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request. Your contributions are welcome!
