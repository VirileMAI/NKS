# api_server.py — FastAPI веб-сервер для чата с моделью (ЛР3, вариант 17)
# Тема: Sentiment Analysis

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------ 1. Инициализация FastAPI ------------------
app = FastAPI(title="NKS Chat API | Sentiment Analysis", version="1.0")

# Монтируем папку static для раздачи HTML, CSS, JS файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------ 2. Глобальная загрузка модели ------------------
# Модель загружается ОДИН РАЗ при старте сервера
model_path = "./my_finetuned_model_lab3"
print("[INFO] Загрузка токенизатора и модели...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()  # Режим инференса
print("[INFO] Модель успешно загружена и готова к запросам.")


# ------------------ 3. Pydantic модели (валидация JSON) ------------------
class ChatMessage(BaseModel):
    """Структура одного сообщения в истории диалога"""
    user: str
    bot: str


class ChatRequest(BaseModel):
    """Структура входящего POST-запроса на /chat"""
    message: str                     # Текущий вопрос пользователя
    history: list[ChatMessage] = []  # История предыдущих реплик


class ChatResponse(BaseModel):
    """Структура ответа сервера"""
    response: str


# ------------------ 4. Обработчик главной страницы ------------------
@app.get("/")
async def read_root():
    """Возвращает index.html при заходе в корень сайта"""
    return FileResponse("static/index.html")


# ------------------ 5. Основной API-эндпоинт ------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Принимает сообщение пользователя и историю диалога.
    Возвращает сгенерированный ответ модели в формате JSON.
    """
    try:
        # --- Шаг 1: Формирование промпта с историей ---
        prompt = ""
        for msg in request.history:
            prompt += f"Вопрос: {msg.user}\nОтвет: {msg.bot}\n\n"
        prompt += f"Вопрос: {request.message}\n\nОтвет:"

        # --- Шаг 2: Токенизация ---
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # --- Шаг 3: Генерация (инференс) ---
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        # --- Шаг 4: Декодирование ---
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not response:
            response = "[Модель не смогла сформулировать ответ]"

        return ChatResponse(response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}")


# ------------------ 6. Точка входа ------------------
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" позволяет подключаться с других устройств в локальной сети
    uvicorn.run(app, host="0.0.0.0", port=8000)
