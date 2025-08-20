# Fine-Tuned GPT-2 for Product Review Generation

## 📌 Overview
This project fine-tunes **GPT-2** to generate realistic product reviews based on structured product context.  
It demonstrates the **full applied AI pipeline**: data preprocessing → model training → evaluation → deployment.

**Goal:** Improve the quality and relevance of generated reviews compared to the original GPT-2 model, with measurable improvement in perplexity and output coherence.

---

## 🛠 Features
- Preprocessing pipeline for structured review datasets of consumers reviews from Amazon
- Fine-tuning GPT-2 with **Hugging Face Transformers**  
- Quantitative evaluation (perplexity)  
- Qualitative comparison of baseline vs fine-tuned output  
- Optional **Streamlit app** for live generation  
- Reproducible training & evaluation scripts

---

## 📂 Project Structure


project-root/
│
├── data_set/ # Example dataset
├── notebook/ # Jupyter notebook for preprocessing, training & evaluation
├── python-script/ # Modular Python script
├── models/ # Saved model checkpoints
├── README.md
├── requirements.txt
└── LICENSE


---

## 📊 Problem Statement
E-commerce platforms require realistic product reviews for:
- Testing recommendation systems  
- Improving model training datasets  
- Generating synthetic data for low-resource product categories  

---

## 📑 Dataset
- **Source:** [[Kaggle](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)]  
- **Format:** CSV with columns:
  - `Product`
  - `Category`
  - `Rating`
  - `Review`
- **Preprocessing Steps:**
  - Remove extra whitespace
  - Merge context as:  
    ```
    Product: <product name>
    Category: <category>
    Rating: <rating>
    Review: <review text>
    ```
  - Tokenize with GPT-2 tokenizer (with pad token set to EOS)

---

## 🧠 Model
- **Base Model:** GPT-2 (`gpt2` from Hugging Face)  
- **Fine-Tuning Task:** Causal Language Modeling (CLM)  
- **Key Hyperparameters:**
  - Epochs: 3
  - Batch Size: 2
  - Learning Rate: 5e-5
  - Weight Decay: 0.01
  - Max Length: 512 tokens

---

## 📈 Results

### Quantitative
| Metric         | Baseline GPT-2 | Fine-Tuned GPT-2 |
|----------------|---------------|------------------|
| Perplexity     | 2.93          | 2.48             |

### Qualitative
**Prompt:**
Product: All-New Fire HD 8 Tablet
Category: electronics, tablets
Rating: 5.0
Review:

**Baseline GPT-2 Output:**
> _[Rating: 4.0 very easy to set up and use. easy to download and navigate. very good product. easy to use. very easy to set up and use. very good product. no need to have special apps needed...]_  

**Fine-Tuned GPT-2 Output:**
> _[i bought this for my wife and daughter-in-law. she loves it. they both love their tablets. it's easy to use, and she has plenty of apps available. she uses it to read, play games and watch movies on it. she also has an ipod and a couple of apps she would probably not have purchased on a tablet...]_

---

## ⚙️ Installation
```bash
git clone https://github.com/nourkarnib/Gen-AI-Opinion-about-products-based-on-AWS-data.git
cd Gen-AI-Opinion-about-products-based-on-AWS-data
pip install -r requirements.txt

## Link to the app

https://huggingface.co/spaces/nourkarnib/fine-tuned-gpt2

## ✨ Future Work

Compare against other LMs (GPT-Neo, LLaMA)

Add RLHF for human-aligned generation

Evaluate with BLEU/ROUGE metrics