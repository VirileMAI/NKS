# NKS Lab 3 — Deployment of Fine-tuned Language Model

**Variant:** 17
**Topic:** Sentiment Analysis
**Model:** Qwen/Qwen2.5-0.5B-Instruct + LoRA adapter (from Lab 2)

## Project Structure

```
lr3/
├── static/
│   └── index.html          # Web chat frontend
├── screenshots/            # Screenshots for the report
├── merge_model.py          # Merge base model + LoRA (run once)
├── run_chat.py             # Console chat
├── api_server.py           # FastAPI web server
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

## Getting the Model

Model weights are not included in the repository due to size.
To reproduce:

1. Download the LoRA adapter from Lab 2 output (`lora_adapter_variant_17/`)
2. Run `python merge_model.py` to merge weights
3. The merged model will be saved to `my_finetuned_model_lab3/`

Alternatively, download the base model directly:
```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); m.save_pretrained('./my_finetuned_model_lab3'); t.save_pretrained('./my_finetuned_model_lab3')"
```

## Running

### Console Chat
```bash
python run_chat.py
```

### Web Server (FastAPI)
```bash
python api_server.py
```
Then open: http://localhost:8000
