# ECG Chagas Detection (PhysioNet Challenge 2025)

From the [`CODE-15% dataset`](https://zenodo.org/records/4916206), we have downloaded the [`exams_part0.zip`](https://zenodo.org/records/4916206/files/exams_part0.zip?download=1) file, the [`exams.csv`](https://zenodo.org/records/4916206/files/exams.csv?download=1) file, and the [`code15_chagas_labels.csv`](https://physionetchallenges.org/2025/data/code15_chagas_labels.zip) label file.

Once downloaded and unzipped, we need to convert the CODE-15% dataset to WFDB format, with the available demographics information and Chagas labels in the WFDB header file, by running

```bash
python prepare_code15_data.py \
    -i code15_hdf5/exams_part0.hdf5 \
    -d code15_hdf5/exams.csv \
    -l code15_hdf5/code15_chagas_labels.csv \
    -o code15_wfdb
```

Each `exam_part` file in the [CODE-15% dataset](https://zenodo.org/records/4916206) contains approximately 20,000 ECG recordings.

> [!TIP]
> After this, we need to extract also the tar.bz2 file (`data.tar.bz2`) which contains the ECG recordings of Chagas patients. Extract it to the `code15_wfdb` directory as to have both the negative and positive samples in the same directory.

## Useful links

* [Challenge website](https://physionetchallenges.org/2025/)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2025)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2025/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
