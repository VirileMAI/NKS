# run_chat.py — Консольный чат с дообученной моделью (ЛР3, вариант 17)
# Тема: Sentiment Analysis

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. Загрузка модели и токенизатора
# ==========================================

model_path = "./my_finetuned_model_lab3"  # Папка с объединённой моделью
print("[INFO] Загрузка токенизатора и модели...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Настройка pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()  # Режим инференса (отключаем Dropout)
print("[INFO] Модель загружена. Введите 'выход' для завершения.\n")

# ==========================================
# 2. Цикл диалога
# ==========================================

chat_history = ""

while True:
    user_input = input("Вы: ").strip()

    if not user_input:
        continue
    if user_input.lower() in ["выход", "exit", "quit"]:
        print("До свидания!")
        break

    # Формирование промпта
    prompt = f"{chat_history}Вопрос: {user_input}\n\nОтвет:"

    # Токенизация
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Декодирование — берём только новые токены
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not response:
        response = "[Модель не смогла сформулировать ответ]"

    print(f"Бот: {response}\n")

    # Обновляем историю
    chat_history += f"Вопрос: {user_input}\nОтвет: {response}\n\n"
