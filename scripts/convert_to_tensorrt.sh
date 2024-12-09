#!/bin/bash

ONNX_MODEL="model.onnx"
OUTPUT_DIR="tensorrt_models"

mkdir -p ${OUTPUT_DIR}

# FP32
echo "Конвертация в FP32..."
trtexec \
    --onnx=${ONNX_MODEL} \
    --saveEngine=${OUTPUT_DIR}/model_fp32.plan \
    --minShapes=INPUT_IDS:1x128,ATTENTION_MASK:1x128 \
    --optShapes=INPUT_IDS:4x128,ATTENTION_MASK:4x128 \
    --maxShapes=INPUT_IDS:8x128,ATTENTION_MASK:8x128

# FP16
echo "Конвертация в FP16..."
trtexec \
    --onnx=${ONNX_MODEL} \
    --saveEngine=${OUTPUT_DIR}/model_fp16.plan \
    --fp16 \
    --minShapes=INPUT_IDS:1x128,ATTENTION_MASK:1x128 \
    --optShapes=INPUT_IDS:4x128,ATTENTION_MASK:4x128 \
    --maxShapes=INPUT_IDS:8x128,ATTENTION_MASK:8x128

# INT8
echo "Конвертация в INT8..."
trtexec \
    --onnx=${ONNX_MODEL} \
    --saveEngine=${OUTPUT_DIR}/model_int8.plan \
    --int8 \
    --minShapes=INPUT_IDS:1x128,ATTENTION_MASK:1x128 \
    --optShapes=INPUT_IDS:4x128,ATTENTION_MASK:4x128 \
    --maxShapes=INPUT_IDS:8x128,ATTENTION_MASK:8x128

# Best
echo "Конвертация в Best..."
trtexec \
    --onnx=${ONNX_MODEL} \
    --saveEngine=${OUTPUT_DIR}/model_best.plan \
    --best \
    --minShapes=INPUT_IDS:1x128,ATTENTION_MASK:1x128 \
    --optShapes=INPUT_IDS:4x128,ATTENTION_MASK:4x128 \
    --maxShapes=INPUT_IDS:8x128,ATTENTION_MASK:8x128

echo "Конвертация завершена. Файлы сохранены в ${OUTPUT_DIR}"