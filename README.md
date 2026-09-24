# High-Performance BERT Fine-Tuning on SQuAD v2

**Authors:** Nikhil Arora and Devanshi Bhavsar
**Course:** High Performance Machine Learning, New York University, Fall 2025

This project studies how systems and training choices change the speed, memory use, and accuracy of BERT-base fine-tuning for extractive question answering. We benchmarked each configuration on SQuAD v2 using a single NVIDIA T4 with a 15-16 GB memory constraint.

The goal was practical: identify configurations that improve throughput or reduce VRAM without hiding the accuracy cost.

## Results

| Configuration | Main change | Throughput (samples/s) | Dev F1 |
|---|---|---:|---:|
| A1 | FP16 AdamW baseline | 43.9 | 76.6 |
| A3 | SDPA + `torch.compile` | 45.3 | 76.9 |
| A4-nb | Dynamic padding | 51.6 | 77.2 |
| B1 | LoRA | 70.4 | 54.9 |
| B3 | Freeze 10 encoder layers | 89.8 | 72.0 |
| C1 | 8-bit AdamW | 46.4 | 76.3 |
| C2 | Gradient checkpointing | 37.1 | 74.5 |
| D1 | AOT-eager backend | 45.5 | 76.4 |

Peak allocated memory from a representative training step:

| Configuration | Peak VRAM | Change from A3 |
|---|---:|---:|
| A3 | 3,087 MiB | Baseline |
| C1 | 2,466 MiB | 20% lower |
| C2 | 2,100 MiB | 32% lower |

Dynamic padding produced the best balanced result: about 17% more throughput than A1 while slightly improving F1. Freezing layers was fastest but gave up accuracy. The 8-bit optimizer preserved most baseline quality while reducing memory, whereas gradient checkpointing saved more memory at a larger speed and accuracy cost.

We also retained A4-b as a negative ablation. Its attempted length-bucketing path introduced answer-span label misalignment and caused accuracy to collapse. Keeping that result made the failure mode explicit instead of silently discarding it.

## Experiments

- **A-series:** baseline, scaled dot-product attention, compilation, and dynamic padding
- **B-series:** LoRA, BitFit-style bias-only tuning, and encoder-layer freezing
- **C-series:** 8-bit optimizer and gradient checkpointing
- **D-series:** AOT-eager versus Inductor compilation
- **Profiling:** peak allocated and reserved CUDA memory after warmup

## Repository Layout

```text
.
|-- reports/
|   |-- HPML_Project_Proposal.pdf
|   |-- HPML_Midpoint_Report.pdf
|   `-- HPML_Final_Report.pdf
|-- scripts/
|   `-- measure_vram_qa.py
|-- transformers/examples/pytorch/question-answering/
|   |-- run_qa.py
|   |-- run_qa_a2_sdpa.py
|   |-- run_qa_a4_sdpa.py
|   |-- run_qa_b1_lora.py
|   |-- run_qa_b2_bitfit.py
|   |-- run_qa_b3_freeze.py
|   |-- run_qa_c1.py
|   |-- run_qa_c2.py
|   |-- run_qa_d1.py
|   |-- trainer_qa.py
|   `-- utils_qa.py
`-- requirements.txt
```

The experiment scripts are adaptations of the Hugging Face question-answering example. Raw SQuAD v2 data, model checkpoints, and generated outputs are intentionally excluded; the dataset is downloaded through `datasets` on first use.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build appropriate for the host system. The 8-bit optimizer experiment also requires a platform supported by `bitsandbytes`.

## Example Run

```bash
cd transformers/examples/pytorch/question-answering

python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad_v2 \
  --version_2_with_negative \
  --do_train --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --fp16 \
  --output_dir outputs/a1-baseline \
  --report_to none
```

The other `run_qa_*.py` files preserve the corresponding experiment-specific modifications. For A4, use `--bucket_mode nobug` for the dynamic-padding path or `--bucket_mode bug` only to reproduce the failed bucketing ablation.

To profile memory after configuring the environment:

```bash
python scripts/measure_vram_qa.py a3
python scripts/measure_vram_qa.py c1
python scripts/measure_vram_qa.py c2
```

## Documentation

The [final report](reports/HPML_Final_Report.pdf) contains the experimental design, hardware setup, full tables, plots, and analysis. The [proposal](reports/HPML_Project_Proposal.pdf) and [midpoint report](reports/HPML_Midpoint_Report.pdf) document how the study evolved.

## Acknowledgements

Built with PyTorch, Hugging Face Transformers, Hugging Face Datasets, PEFT, and SQuAD v2. Adapted Hugging Face source files retain their original license headers.
