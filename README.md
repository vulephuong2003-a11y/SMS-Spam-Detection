# SMS Spam Detection using Transformer and Hybrid Model

## Giới thiệu
Tiểu luận tái hiện và nâng cấp bài báo:
> **ExplainableDetector: Exploring Transformer-based Language Modeling 
> Approach for SMS Spam Detection with Explainability Analysis**  
> Uddin et al., 2024

## Cấu trúc project

| Thư mục | Nội dung |
|---------|----------|
| `PhucLucA_Thuc_Nghiem_Mo_Hinh_Goc/` | Thực nghiệm lại mô hình gốc |
| `PhucLucB_Mo_Hinh_Nang_Cap/` | Mô hình đề xuất mới |

## Phụ lục A — Thực nghiệm mô hình gốc

| Notebook | Nội dung |
|----------|----------|
| `Stage1_ML.ipynb` | ML Models: XGB, SVM, KNN, RF |
| `Stage2_Transformer.ipynb` | DistilBERT + RoBERTa fine-tuning |
| `Stage3_XAI.ipynb` | LIME + Transformers Interpret |

## Phụ lục B — Mô hình đề xuất mới

**Hybrid RoBERTa-CNN-BiLSTM-Attention**

Kiến trúc mới đề xuất:RoBERTa (Backbone) → CNN 1D → BiLSTM → Self-Attention → Classifier
| Thay đổi | Lý do |
|----------|-------|
| CNN 1D | Bắt local n-gram spam patterns |
| BiLSTM | Ngữ cảnh 2 chiều toàn câu |
| Self-Attention | XAI nội tại — không cần LIME/SHAP |

## Kết quả

| Model | Accuracy |
|-------|----------|
| XGB | 98.34% |
| SVM | 99.59% |
| RF | 99.38% |
| DistilBERT | 99.95% |
| RoBERTa | 99.79% |
| **Hybrid (Proposed)** | **[điền số thực tế]%** |

## Pre-trained Models
File model (.pt) lưu tại Google Drive:  
[Link Drive]

## Môi trường chạy
- Google Colab (GPU T4)
- Python 3.10
- PyTorch 2.x
- Transformers 4.35.0
