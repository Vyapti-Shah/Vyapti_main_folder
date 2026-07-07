# DeepFake / AI-Generated Image Detector

This project contains a Python script (`detect.py`) that uses an **EfficientNet-B0** PyTorch model to classify an image as either "Real" or "AI Generated" and applies a watermark text accordingly at the top-left corner of the image.

## Requirements

Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

## Model Weights Download Links

You will need a `.pt` or `.pth` model weights file to run the script. This script expects an EfficientNet-B0 model trained for binary classification (Real vs Fake).

Here are some places where you can download PyTorch weights for EfficientNet-B0 based DeepFake/AI Image detectors:

1. **General AI Generated Image Detectors**:
   - Some models are available on Hugging Face Model Hub. You can search for models and download their `pytorch_model.bin` or `model.pt` files.
   
2. **Face-based DeepFake Models**:
   - [Xicor9/efficientnet-b0-ffpp-c23](https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/tree/main) (Download the `model.pth` file). Note that this is trained specifically on faces, so it may not perform well on general AI-generated landscapes.
   - [divyanshu-chauhan-7786/deepfake_image](https://huggingface.co/divyanshu-chauhan-7786/deepfake_image)

*Note: If you have your own `model.pt` trained on PyTorch EfficientNet-B0, you can use that directly!*

## How to use

Run the script by providing the input image, the desired output path, and the model weights file.

```bash
python main.py -i <path_to_input_image> -o <path_to_output_image> -m <path_to_model.pt>
```

### Examples:

```bash
# Example assuming you downloaded model.pth
python main.py -i sample.jpg -o result.jpg -m model.pth
```

By default, the script assumes the model outputs **Class 1** for Fake/AI-generated images. If your specific model uses **Class 0** for Fake, you can pass `--fake_class_index 0`:

```bash
python main.py -i sample.jpg -o result.jpg -m model.pth --fake_class_index 0
```

## Watermark

Depending on the model's prediction, the script will add a watermark to the top-left of the image:
- If Real: **"REAL"** (Green text)
- If Fake: **"AI Generated"** (Red text)
