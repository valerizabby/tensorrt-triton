# Используем официальный образ NVIDIA TensorRT
FROM nvcr.io/nvidia/tensorrt:22.09-py3

# Устанавливаем зависимости для работы с Git и Python
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean

# Создаем рабочую директорию внутри контейнера
WORKDIR /workspace

# Переменная окружения для работы с GPU
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Клонируем проект из GitHub
ARG REPO_URL=https://github.com/valerizabby/tensorrt-triton
RUN git clone ${REPO_URL} project

# Устанавливаем зависимости Python, если нужны
RUN pip install --no-cache-dir -r project/requirements.txt

# Указываем volume для обмена файлами между хостом и контейнером
VOLUME /workspace/project

# Переходим в директорию проекта
WORKDIR /workspace/project

# Указываем скрипт, который будет запускаться при старте контейнера
ENTRYPOINT ["python", "scripts/convert_to_tensorrt.sh"]