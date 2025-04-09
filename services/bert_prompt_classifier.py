from transformers import BertTokenizer, BertForSequenceClassification
import torch
import logging

logger = logging.getLogger(__name__)

class BertPromptClassifier:
    def __init__(self, model_path: str = 'nicksedov/rubert-tiny2-classifier'):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Переводим модель в режим инференса
        logger.info("BERT classifier initialized")

    def is_image_request(self, text: str) -> bool:
        # Предобработка и токенизация текста
        inputs = self._preprocess_text(text)
        
        # Выполнение предсказания
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Получение предсказания
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return bool(prediction)

    def _preprocess_text(self, text: str) -> dict:
        # Токенизация с учетом требований модели BERT
        return self.tokenizer(
            text,
            padding=True,         # Автоматическое дополнение до максимальной длины в батче
            truncation=True,      # Обрезка длинных текстов
            max_length=512,       # Максимальная длина согласно спецификации BERT
            return_tensors='pt'   # Возврат тензоров PyTorch
        )