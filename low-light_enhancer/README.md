# Low-Light Video Enhancement Pipeline

This repository is focused on enhancing low-light and underexposed video footage. Its primary goal is luminance-aware restoration through Contrast Limited Adaptive Histogram Equalization (CLAHE) and global contrast boosting.

## Key Features

- Low-light luminance correction using CLAHE
- Global contrast boosting
- Forensic watermark indicating low-light enhancement

## Project Structure

- `main.py` - Primary command-line entry point for the modular enhancement pipeline
- `forensic_enhancer.py` - Alternative standalone single-file pipeline implementation
- `config.py` - Central enhancement configuration and default parameters
- `pipeline/` - Modular pipeline stages and orchestration
- `processing/` - Low-light frame enhancement components
- `video/` - Video frame extraction and compilation utilities
- `utils/` - Helper utilities such as progress tracking

## How It Works

The enhancement pipeline runs in three main stages:

1. **Frame Extraction**
   - Extracts raw frames from the input video

2. **Frame Enhancement**
   - Applies CLAHE and contrast boost frame-by-frame
   - Adds a forensic watermark ("ENHANCED (LOW LIGHT)")

3. **Reassemble Video**
   - Compiles enhanced frames back into an output video while preserving the original FPS

## Usage

Run the pipeline from the repository root.

```bash
python main.py -i input.mp4 -o output.mp4
```

Or using the standalone script:

```bash
python forensic_enhancer.py -i input.mp4 -o output.mp4
```

## Dependencies

The code uses the following libraries:

- `opencv-python`
- `numpy`
- `tqdm`

## License

No license file is provided in this repository. Use the code according to your own risk and environment policies.
