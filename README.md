**We sincerely appreciate your time reviewing this code, especially during this era of surging submission volumes. **

# Setup

```python
conda create -n vdahm python==3.11.9
conda activate vdahm
pip install -r requirements.txt
```

# Dataset

For evaluation, we employ four datasets: Mantis-Eval, BLINK, MuirBench, and Qbench2.

Mantis-Eval: [TIGER-Lab/Mantis-Eval · Datasets at Hugging Face](https://huggingface.co/datasets/TIGER-Lab/Mantis-Eval)

BLINK: [BLINK-Benchmark/BLINK · Datasets at Hugging Face](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)

MuirBench: [MUIRBENCH/MUIRBENCH · Datasets at Hugging Face](https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH)

Qbench2: [q-future/q-bench2 at main](https://huggingface.co/datasets/q-future/q-bench2/tree/main)

# Configuration Parameters

### Model & Data Paths

- `--json_file_path`(str): Path to the evaluation dataset JSON file 

- `--model_path`(str): Path to LVLMs weights
- `--image_fix`(str): Base directory path where image files are stored
- `--output_json_path`:(str): Path where evaluation results will be saved in JSON format

### VDAHM Parameters

- `--vdahm_aug_ratio`(float): VDAHM augmentation ratio
- `--vdahm_layers`(int): Number of layers to apply VDAHM
- `--vdahm_layer1`(bool): Whether to include layer 1 in VDAHM
- `--vdahm_filter`(bool): Whether to apply VDAHM filtering

# Evaluation

To reproduce **Qbench2** benchmark results on Qwen2.5_VL, run:

```python
python qbench2.py
```

# GPU Requirement

The experiment has been validated on NVIDIA RTX A5000 (24GB)

