# merge_model.py — Объединение базовой модели + LoRA-адаптера из ЛР2
# Запускать ОДИН РАЗ перед началом ЛР3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# Параметры — ЗАМЕНИТЬ на свои пути!
# ==========================================

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"       # Базовая модель из ЛР2
adapter_path = "./lora_adapter_variant_17"              # Папка с LoRA-адаптером из ЛР2
output_path = "./my_finetuned_model_lab3"               # Куда сохранить объединённую модель

# ==========================================
# 1. Загрузка базовой модели
# ==========================================
print("[INFO] Загрузка базовой модели...")
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# ==========================================
# 2. Загрузка LoRA-адаптера
# ==========================================
print("[INFO] Загрузка LoRA-адаптера...")
model = PeftModel.from_pretrained(model, adapter_path)

# ==========================================
# 3. Объединение весов и сохранение
# ==========================================
print("[INFO] Объединение весов (merge_and_unload)...")
merged_model = model.merge_and_unload()

merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"[OK] Объединённая модель сохранена в {output_path}")
print("[INFO] Теперь можно запускать run_chat.py и api_server.py")
