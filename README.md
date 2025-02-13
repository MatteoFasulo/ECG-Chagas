# ECG Chagas Detection (PhysioNet Challenge 2025)

## Introduction

Chagas disease is a parasitic infection caused by Trypanosoma cruzi, primarily transmitted by triatomine insects. It can also spread through blood transfusions, organ transplants, and congenital transmission. The disease progresses in two phases: an acute phase, which often presents with mild or nonspecific symptoms such as fever, swollen lymph nodes, and localized swelling at the bite site, and a chronic phase, which develops over time. While most individuals in the chronic stage remain asymptomatic, 30-40% experience severe complications, particularly affecting the heart. Early treatment with antiparasitic drugs is effective, but its success diminishes as the infection advances.

Named after the Brazilian physician Carlos Chagas, who discovered it in 1909, the disease predominantly manifests in its chronic form, usually appearing 10 to 30 years after the initial infection. While the acute phase is generally mild or asymptomatic, the chronic stage can lead to serious cardiac and digestive complications. Diagnosis is typically made by detecting the parasite in the blood or identifying antibodies through serological tests.

Electrocardiography (ECG) plays a crucial role in identifying Chagas-related heart abnormalities. In the chronic stage, Chagas cardiomyopathy can result in progressive heart damage, conduction disorders such as atrioventricular blocks, ventricular arrhythmias, and dilated cardiomyopathy. In severe cases, it can lead to sudden cardiac death due to fatal arrhythmias like ventricular fibrillation. However, the indeterminate chronic form does not show ECG abnormalities, making early detection difficult.

## Dataset

From the [`CODE-15% dataset`](https://zenodo.org/records/4916206), we have created a balanced subset of ECG records as to balance seropositive and seronegative cases. [The dataset is available in Kaggle](https://www.kaggle.com/datasets/matteofasuloo/code15-ecg-chagas-balanced).

At the end the the following folder structure should be created:

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
├──  ....
```

> [!TIP]
> The `code15_wfdb` directory contains the ECG records in WFDB format.

## Requirements

The project was developed using Python 3.10.0. The required packages are listed in the `requirements.txt` file. Alternatively, you can install the required packages using the `pyproject.toml` file with your preferred package manager.

For the standard `pip` package manager, you can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Jupyter Notebook Slideshow

The Jupyter Notebook slideshow can be generated using the following command:

```python
jupyter nbconvert --to slides notebook.ipynb --post serve
```

In this way, the notebook will be converted to a slideshow and opened in the default browser.

### NBconvert Reveal.js issue

As of the writing (Feb 2025), there is an [issue with the `nbconvert` package that automatically sets the cells slide type to `-` instead of `skip`](https://github.com/jupyter/nbconvert/issues/2069). In this way each cell must be manually set to `skip` to avoid displaying the cell content in the slideshow in a concatenated way.

The script `fix_metadata.py` can be used to automatically set the cell type to `skip` for all cells in the notebook. Moreover, it will set the first level headings to `slide` and the second/third level headings to `subslide`.

The syntax to run the script is the following:

```bash
python fix_metadata.py notebook.ipynb
```

## Useful links

* [Challenge website](https://physionetchallenges.org/2025/)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2025)
* [Kaggle Dataset](https://www.kaggle.com/datasets/matteofasuloo/code15-ecg-chagas-balanced)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2025/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
