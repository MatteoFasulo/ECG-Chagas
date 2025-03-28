# ECG Chagas Detection

## Introduction

Chagas disease, caused by the parasite *Trypanosoma cruzi*, is primarily transmitted through triatomine insect bites but can also spread via blood transfusions, organ transplants, and congenital transmission. The disease progresses in two stages: the **acute phase**, characterized by mild symptoms such as fever, swollen lymph nodes, and localized swelling at the bite site, and the **chronic phase**, which can develop years after the initial infection. While the acute phase is often asymptomatic or mild, around 30-40% of individuals in the chronic phase develop severe complications, especially affecting the heart.

The chronic form of the disease can result in **Chagas cardiomyopathy**, which manifests as **heart damage**, **conduction disorders** (like atrioventricular blocks), **ventricular arrhythmias**, and **dilated cardiomyopathy**. In extreme cases, this can lead to **sudden cardiac death** from fatal arrhythmias like **ventricular fibrillation**. However, the indeterminate chronic form often shows no **ECG abnormalities**, making early diagnosis more challenging.

This project focuses on **ECG-based detection** of Chagas disease in the chronic stage, where detecting **heart-related abnormalities** is key to understanding the severity of the disease.

## **Dataset**

We use the [`CODE-15% dataset`](https://zenodo.org/records/4916206), which contains **ECG records** of individuals, both **seropositive** and **seronegative** for Chagas disease. A **balanced subset** of this dataset has been created to ensure an equal representation of both positive and negative cases.

The dataset is available for download on [Kaggle](https://www.kaggle.com/datasets/matteofasuloo/code15-ecg-chagas-balanced). The directory structure for the dataset is as follows:

```bash
data/
├── signals_features.csv
code15_hdf5/
├── code15_chagas_labels.csv
├── exams.csv
code15_wfdb/
├── 10099.dat
├── 10099.hea
├── 10686.dat
├── 10686.hea
├── .... (additional records)
```

> **Note**: The `code15_wfdb` folder contains ECG data in **WFDB** format.

## Results

| Model | Accuracy | ROC-AUC | F1 Score |
|-------|----------|---------|----------|
| Se-ResNet18 | 0.75 | 0.84 | 0.75 |
| Transformer-Hand | 0.74 | 0.83 | 0.75 |
| Transformer | 0.73 | 0.82 | 0.71 |
| XGBoost | 0.70 | 0.78 | 0.71 |
| XGBoost-SHAP | 0.70 | 0.78 | 0.71 |
| KNN | 0.69 | 0.77 | 0.70 |
| LogisticRegression | 0.68 | 0.74 | 0.67 |


## **Requirements**

This project was developed using **Python 3.10.0**. The required dependencies are listed in the `requirements.txt` file. To install the dependencies, use the following command:

```bash
pip install -r requirements.txt
```

Alternatively, if you're using a package manager such as `poetry`, you can install the dependencies via the `pyproject.toml` file.

## **Jupyter Notebook Slideshow**

The project provides a **Jupyter Notebook** for analysis and model training. To convert the notebook to a slideshow and view it in your browser, run the following command:

```bash
jupyter nbconvert --to slides notebook.ipynb --post serve
```

### **NBconvert Reveal.js Issue**

As of **February 2025**, there's an issue with the `nbconvert` package that causes each cell to be set to `-` (default) rather than `skip` in slides. To fix this, use the script `fix_metadata.py`, which automatically sets the slide type for all cells and adjusts the heading levels.

To run the fix script, use the following command:

```bash
python fix_metadata.py notebook.ipynb
```

## **Useful Links**

- **[Challenge Website](https://physionetchallenges.org/2025/)**: Access more information on the challenge.
- **[Evaluation Code](https://github.com/physionetchallenges/evaluation-2025)**: For evaluating the model's performance.
- **[Kaggle Dataset](https://www.kaggle.com/datasets/matteofasuloo/code15-ecg-chagas-balanced)**: Download the ECG dataset.
- **[FAQ (2025 Challenge)](https://physionetchallenges.org/2025/faq/)**: Frequently asked questions about this year’s challenge.
- **[General FAQ](https://physionetchallenges.org/faq/)**: FAQs related to previous challenges.

## **License**

This project is licensed under the **MIT License**. Please refer to the [LICENSE](LICENSE) file for more details.