# FLAN-T5 Mathematics Reasoning with LoRA

I fine-tuned **Google FLAN-T5 Base** on **GSM8K** to see whether a relatively small encoder–decoder model could learn to produce worked mathematics solutions. I trained for **five epochs**, but the model still struggled with the questions I tested. This repository records the setup, evaluation approach, and what I learned.

## What I used

- Base model: [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base)
- Dataset: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), `main` configuration
- LoRA: rank 8, alpha 16, dropout 0.05, targeting `q` and `v`
- Optimizer/training: Hugging Face `Seq2SeqTrainer`, learning rate `1e-4`, effective batch size 8
- Training: 3 initial epochs, followed by 2 continuation epochs
- Trainable parameters reported in original run: **884,736 (0.3561%)**

## Repository contents

```text
flan-t5-math-reasoning-lora/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── flan_t5_math_lora.ipynb
```

## Running the notebook

1. Set up a Python environment with CUDA-enabled PyTorch if using an NVIDIA GPU.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `notebooks/flan_t5_math_lora.ipynb` in Jupyter or VS Code.
4. Run the preparation cells, then uncomment and run the first training stage (3 epochs).
5. Run the continuation stage (2 epochs) **only after the first stage finishes**.
6. Run the generation and evaluation cells to inspect the results.

Training takes time and downloads the model and dataset. Trained weights and checkpoints are intentionally excluded from this repository.

## Results and limitations

I completed five epochs but still observed incorrect answers in manual testing. The original notebook includes two recorded loss pairs beside the continuation code:

| Recorded continuation epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 5.611812 | 1.123578 |
| 2 | 5.541370 | 1.120934 |

These are historical notes, **not a verified full five-epoch training log**. The original notebook contains commented-out continuation code, so this repository does not claim to reconstruct every action from the completed run. The cleaned notebook is a readable reproduction recipe and has not been rerun end-to-end.

I did **not** measure reliable accuracy on the full GSM8K test set. The failure to solve the questions I tried should not be described as a measured 0% test accuracy.

## Takeaway

Fine-tuning a model to imitate step-by-step solutions does not guarantee reliable mathematical reasoning. This experiment helped me learn how LoRA works with an encoder–decoder model, how to mask target padding, and why generated answers need to be evaluated beyond training loss.

## Acknowledgements

The base model is from Google, and the GSM8K dataset is from OpenAI. See their model and dataset pages for applicable terms.
