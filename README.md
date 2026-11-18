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

# BibTex

```
@inproceedings{liang-etal-2026-vdahm,
    title = "{VDAHM}: Vision-Delimiter-aware Attention Head Modulation for Enhancing Multi-Image Understanding in Large Vision-Language Models",
    author = "Liang, Zhibiao  and
      Cheng, Junyu  and
      Zeng, Ziqian",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.125/",
    doi = "10.18653/2026.findings-emnlp.125",
    pages = "2542--2561",
    ISBN = "979-8-89176-462-0",
    abstract = "Although Large Vision-Language Models (LVLMs) excel at single-image understanding tasks, their performance often degrades significantly when processing multi-image inputs due to unintended cross-image information leakage, in which visual content from different images is erroneously fused and image-specific contextual information is lost. Current approaches attempt to mitigate this by inserting special vision delimiter tokens between images during input formatting to help models distinguish image boundaries and preserve context. However, our analysis reveals that such delimiters alone yield limited gains because models do not sufficiently perceive or respond to these boundary signals. To address this, we propose Vision-Delimiter-aware Attention Head Modulation (VDAHM), a training-free method that modulates the output of delimiter-sensitive attention heads, reorienting the model{'}s internal attention flow to strengthen its structural perception of boundary signals. Extensive evaluations across multiple benchmarks, including Mantis, BLINK, MuirBench, and Qbench2, demonstrate that VDAHM consistently improves LVLMs performance with virtually no additional computational cost. The source code is available at [https://github.com/ZeroNLP/VDAHM](https://github.com/ZeroNLP/VDAHM)."
}
```

