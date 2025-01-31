# ECG Chagas Detection (PhysioNet Challenge 2025)

From the [`CODE-15% dataset`](https://zenodo.org/records/4916206), we have downloaded the [`exams_part0.zip`](https://zenodo.org/records/4916206/files/exams_part0.zip?download=1) file, the [`exams.csv`](https://zenodo.org/records/4916206/files/exams.csv?download=1) file, and the [`code15_chagas_labels.csv`](https://physionetchallenges.org/2025/data/code15_chagas_labels.zip?download=1) label file.

However, we will provide a convenient way of getting the data by just extracting the tar gzip files (previously splitted into chunks of 250MB).

```bash
cat data.tar.gz.* | tar xzvf -
```

At the end of this process, we will have the following directory structure:

```bash
code15_wfdb/
├── 10099.dat
├── 10099.hea
├── 10686.dat
├── 10686.hea
```

> [!TIP]
> The `code15_wfdb` directory contains the ECG records in WFDB format.

## Useful links

* [Challenge website](https://physionetchallenges.org/2025/)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2025)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2025/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
