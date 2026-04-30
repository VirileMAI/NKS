import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─────────────────────────────────────────────
# ШАГ 1. Объединяем базовую модель + LoRA-адаптер
# ─────────────────────────────────────────────
print("Загрузка базовой модели и LoRA-адаптера...")

base_model_name = "t-bank-ai/ruDialoGPT-small"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

adapter_path = "./lora_adapter_fully_connected"  # Укажите ваш путь!
model = PeftModel.from_pretrained(model, adapter_path)

# "Вшиваем" адаптер и сохраняем объединённую модель
print("Объединение весов (merge_and_unload)...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./my_finetuned_model_lab3")
tokenizer.save_pretrained("./my_finetuned_model_lab3")
print("Модель сохранена в ./my_finetuned_model_lab3")

# ─────────────────────────────────────────────
# ШАГ 2. Загружаем объединённую модель для инференса
# ─────────────────────────────────────────────
model_path = "./my_finetuned_model_lab3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Настройка технического токена
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()  # Режим инференса (отключает Dropout)

# ─────────────────────────────────────────────
# ШАГ 3. Функция генерации ответа
# ─────────────────────────────────────────────
def generate_response(user_input: str, chat_history: str) -> tuple[str, str]:
    """
    Генерирует ответ модели на вопрос пользователя.

    Args:
        user_input:   текущий вопрос пользователя
        chat_history: накопленная история диалога (строка)

    Returns:
        (ответ модели, обновлённая история)
    """
    # Формируем промпт в формате ruDialoGPT
    if not chat_history:
        prompt = f"@@ПЕРВЫЙ@@{user_input}@@ВТОРОЙ@@"
    else:
        prompt = f"{chat_history}@@ПЕРВЫЙ@@{user_input}@@ВТОРОЙ@@"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,      # Максимальная длина ответа в токенах
            do_sample=True,          # Включаем случайность
            top_k=50,                # Топ-50 наиболее вероятных токенов
            top_p=0.9,               # Ядерная (nucleus) фильтрация
            temperature=0.8,         # «Температура» распределения
            repetition_penalty=1.2,  # Штраф за повторы слов
            no_repeat_ngram_size=3,  # Запрет повтора 3-грамм
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Декодируем только новые токены (после промпта)
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Обновляем историю диалога
    updated_history = prompt + response

    return response, updated_history


# ─────────────────────────────────────────────
# ШАГ 4. Интерактивная чат-сессия
# ─────────────────────────────────────────────
print("\nМодель готова. Введите 'выход' для завершения.\n")

chat_history = ""

while True:
    user_input = input("Вы: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ("выход", "exit", "quit"):
        print("Сессия завершена.")
        break

    response, chat_history = generate_response(user_input, chat_history)
    print(f"Модель: {response}\n")