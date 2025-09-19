#!/bin/bash

# Скрипт для установки зависимостей с правильными версиями NumPy

echo "Installing dependencies with NumPy compatibility fix..."

# Сначала удаляем существующие версии проблемных пакетов
pip uninstall -y numpy torch transformers tokenizers

# Устанавливаем NumPy 1.x (совместимую версию)
pip install "numpy>=1.24.0,<2.0.0"

# Устанавливаем PyTorch с совместимой версией NumPy
pip install torch==2.2.2

# Устанавливаем transformers и tokenizers
pip install transformers==4.44.2 tokenizers==0.19.1

# Устанавливаем остальные зависимости
pip install -r requirements.txt

echo "Dependencies installed successfully!"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
