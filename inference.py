import argparse
import os

# 国内网络访问 HuggingFace 较慢时自动使用镜像；若你已设置 HF_ENDPOINT 则保留你的设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "microsoft/resnet-50"  # 预训练 ResNet 模型
IMAGE_SIZE = (224, 224)             # ResNet 的输入尺寸
BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-samples", type=int, default=10000,
        help="只取测试集前多少张图片，默认 10000（= 全部测试集）",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"loading pretrained ResNet model: {MODEL_NAME} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("loading MNIST dataset ...")
    # 用带命名空间的规范仓库名 ylecun/mnist
    test = load_dataset("ylecun/mnist", split="test").select(range(args.max_samples))

    correct = 0
    total = 0
    for start in range(0, len(test), BATCH_SIZE):
        batch = test[start:start + BATCH_SIZE]

        # MNIST 是 28x28 灰度图：先转成 RGB 三通道，再 resize 到 ResNet 需要的 224x224
        images = [img.convert("RGB").resize(IMAGE_SIZE) for img in batch["image"]]
        labels = batch["label"]

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu()

        correct += (preds == torch.tensor(labels)).sum().item()
        total += len(labels)
        print(f"progress: {total}/{len(test)}")

        if start == 0:
            # 打印前 5 个样例，看看模型把数字识别成了什么
            for j in range(min(5, len(labels))):
                pred_class = preds[j].item()
                print(f"  sample {j}: label={labels[j]} -> "
                      f"pred={pred_class} ({model.config.id2label[pred_class]})")

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
