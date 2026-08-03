# Illusion Breaker

This project runs the batch image pipeline using YAML configuration files.

## Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run with the digit configuration

Use the digit config to process images from the digits input folder:

```bash
python run_batch.py --config config_digits.yaml
```

This will read images from `./data/input/digits` and write results to `./data/output/digits`.

## Run with the text configuration

Use the text config to process images from the text input folder:

```bash
python run_batch.py --config config_text.yaml
```

This will read images from `./data/input/text` and write results to `./data/output/text`.

## Note about `config.yaml`

The file `config.yaml` is currently incorrect and should not be used for running the pipeline.

Please use either:

- `config_digits.yaml`
- `config_text.yaml`
