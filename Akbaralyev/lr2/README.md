# ЛР2. Адаптация языковой модели с помощью QLoRA

| Параметр | Значение |
|----------|----------|
| Студент | Акбаралыев А.А. |
| Группа | М3О-419Бк-22 |
| Вариант | 5 |
| Тема | Тьютор по LSTM и управлению памятью |

## Структура

```
lr2/
├── main.ipynb                     # Основной ноутбук (код + результаты)
├── lora_adapter_variant_5/        # Обученный LoRA-адаптер
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── README.md
```

Датасет (`corpus_variant_5/`) находится в `../lr1/` — повторно не дублируется.

## Модели

В ноутбуке предусмотрено обучение **5 моделей** через QLoRA / LoRA:

| Модель | Размер | Метод | Среда |
|--------|--------|-------|-------|
| ai-forever/rugpt3small_based_on_gpt2 | 125M | LoRA | MPS / CPU |
| google/gemma-2-2b-it | 2.6B | QLoRA 4-bit | Colab (CUDA) |
| microsoft/phi-2 | 2.7B | QLoRA 4-bit | Colab (CUDA) |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | QLoRA 4-bit | Colab (CUDA) |
| unsloth/gemma-2-2b-it-bnb-4bit | 2.6B | QLoRA 4-bit | Colab (CUDA) |

В данном репозитории лежит адаптер для **rugpt3small** (обучен локально на MPS за 21 сек).
Адаптеры для остальных моделей обучаются на Google Colab с GPU.

## Результаты обучения (rugpt3small)

- **Параметры:** 126,853,632 всего, 1,622,016 обучаемых (1.28%)
- **LoRA:** r=16, alpha=32, target_modules=[c_attn, c_proj]
- **Final loss:** 4.7469
- **Время:** 21 сек (Apple M4 Pro, MPS)
- **Размер адаптера:** 11.1 KB

## Воспроизведение

```bash
pip install torch transformers datasets peft trl matplotlib
jupyter notebook main.ipynb
```
