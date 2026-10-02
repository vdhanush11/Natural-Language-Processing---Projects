# Model directory

You can place your downloaded Hugging Face `facebook/nllb-200-distilled-600M` directory here.

Recommended:

```text
models/
└── nllb-200-distilled-600M/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors   (or pytorch_model.bin)
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── sentencepiece.bpe.model
    └── special_tokens_map.json
```

Then set:

```text
MODEL_DIR=./models/nllb-200-distilled-600M
```

The large model weights are intentionally not bundled into the source ZIP. This keeps the application package manageable. The backend supports either this local directory or automatic Hugging Face downloading.
