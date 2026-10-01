# Architecture

```text
                    ┌───────────────────────────────┐
                    │ Browser                       │
                    │ HTML + CSS + JavaScript       │
                    └───────────────┬───────────────┘
                                    │ REST/JSON
                    ┌───────────────▼───────────────┐
                    │ FastAPI                       │
                    │ /api/health                    │
                    │ /api/models                    │
                    │ /api/predict                   │
                    │ /api/generate                  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │ ModelService                   │
                    │ lazy loading + preprocessing   │
                    └───────┬───────────┬───────────┘
                            │           │
             ┌──────────────┘           └────────────────┐
             ▼                                           ▼
    ┌──────────────────┐                        ┌──────────────────┐
    │ TensorFlow/Keras │                        │ PyTorch/Transform │
    │ RNN / LSTM / GRU │                        │ DistilGPT-2       │
    └──────────────────┘                        └──────────────────┘
```

## RNN inference

1. Normalize raw input.
2. Convert words with the supplied tokenizer.
3. Keep the last 10 token IDs.
4. Right-pad with zero when fewer than 10 tokens are present.
5. Run the selected Keras model.
6. Convert the 10,000-class softmax into top-k word predictions.
7. Feed the selected argmax word back into the sequence for iterative generation.

## OOV

Unknown words are mapped by the supplied tokenizer to `<OOV>`. This avoids a hard failure for unseen vocabulary.

## GPT inference

If a local trained DistilGPT-2 checkpoint exists, the app uses its local tokenizer and causal LM. No online download is attempted by the app, preventing accidental substitution of a different checkpoint.
