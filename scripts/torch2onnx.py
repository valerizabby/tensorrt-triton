# torch2onnx.py

import torch
from transformers import AutoModel, AutoTokenizer
from torch import nn

# Параметры модели
MODEL_NAME = "../ruBert-base"  # Локальная директория с моделью
HIDDEN_DIM = 768  # Размерность скрытого слоя (зависит от config.json вашей модели)
OUTPUT_DIM = 128  # Размерность эмбеддингов


class TransformerONNX(nn.Module):
    def __init__(self):
        super(TransformerONNX, self).__init__()
        self.transformer = AutoModel.from_pretrained(MODEL_NAME)
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        embeddings = self.fc(last_hidden_state)
        return embeddings

# Создание и экспорт модели в ONNX
def main():
    # Создаем модель
    model = TransformerONNX()
    model.eval()

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Пример текста
    text = "Это пример текста для тестирования модели."
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Экспорт модели в ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "model.onnx",
        input_names=["INPUT_IDS", "ATTENTION_MASK"],
        output_names=["EMBEDDINGS"],
        dynamic_axes={
            "INPUT_IDS": {0: "batch_size"},
            "ATTENTION_MASK": {0: "batch_size"},
            "EMBEDDINGS": {0: "batch_size"},
        },
        opset_version=19,
    )
    print("Модель успешно экспортирована в ONNX!")

if __name__ == "__main__":
    main()