# tensorrt-triton

## Подготовка окружения
С torch какие-то проблемы на других версиях, поэтому я пользуюсь 3.10. Для переключения между версиями использую pyenv:

```bash
#   смена используемой версии
    pyenv global 3.10.0
    
#   конфигурация shell
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    
#    применить изменения
    source ~/.zshrc
```

И создаем виртуалку как обычно

```bash
    python3.10 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

## Git LFS
Расширение для git для работы с big data. Установка

```bash
    brew install git-lfs
```

Инициализация

```bash
    git lfs install
    git clone https://huggingface.co/ai-forever/ruBert-base
```

## TensorRT

Запустим контейнер, чтобы в нем конвертнуть модели