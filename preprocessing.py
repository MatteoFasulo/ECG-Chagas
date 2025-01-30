import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from helper_code import find_records, get_sampling_frequency, load_header, load_signals
from joblib import Parallel, delayed
import neurokit2 as nk

def check_interval(waves_signals: dict) -> pd.DataFrame:
    # Define the relevant keys (also in the order they should be)
    keys = ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
    # Create a dataframe with the data from the dictionary only for the relevant keys
    df_waves = pd.DataFrame({key: waves_signals[key] for key in keys})
    rows = df_waves.shape[0]
    mask = np.zeros(rows)
    for i in range(1, rows):
        start_interval = df_waves.loc[i-1, 'ECG_R_Peaks']
        end_interval = df_waves.loc[i, 'ECG_R_Peaks']
        sliced_df = df_waves[df_waves.isin([start_interval, end_interval]).any(axis=1)]
        # if the sliced df has no missing values, then the interval is correct and the mask should be 1
        first_condition = sliced_df[keys].isnull().sum().sum() == 0
        second_condition = keys == sliced_df.iloc[0].sort_values(ascending=True).index.tolist()
        if first_condition and second_condition:
            mask[i] = 1
    return df_waves[mask == 1]

def ecg_signal_features(row: pd.Series, frequency: int = 400, milliseconds: bool = True) -> dict:
    ECG_P_Peaks, ECG_P_Onsets, ECG_P_Offsets, ECG_Q_Peaks, ECG_S_Peaks, ECG_T_Peaks, ECG_T_Onsets, ECG_T_Offsets, ECG_R_Peaks = row
    # Compute the P wave duration
    P_wave_duration = (ECG_P_Offsets - ECG_P_Onsets)
    # Compute the PR interval
    PR_interval = (ECG_Q_Peaks - ECG_P_Onsets)
    # Compute the PR segment
    PR_segment = (ECG_Q_Peaks - ECG_P_Offsets)
    # Compute the QRS duration
    QRS_duration = (ECG_S_Peaks - ECG_Q_Peaks)
    # Compute the QT interval
    QT_interval = (ECG_T_Offsets - ECG_Q_Peaks)
    # Compute the ST segment
    ST_segment = (ECG_T_Onsets - ECG_S_Peaks)
    
    if milliseconds:
        P_wave_duration = P_wave_duration * 1000 / frequency
        PR_interval = PR_interval * 1000 / frequency
        PR_segment = PR_segment * 1000 / frequency
        QRS_duration = QRS_duration * 1000 / frequency
        QT_interval = QT_interval * 1000 / frequency
        ST_segment = ST_segment * 1000 / frequency

    else:
        P_wave_duration = P_wave_duration / frequency
        PR_interval = PR_interval / frequency
        PR_segment = PR_segment / frequency
        QRS_duration = QRS_duration / frequency
        QT_interval = QT_interval / frequency
        ST_segment = ST_segment / frequency

    return {
        'P_wave_duration': P_wave_duration,
        'PR_interval': PR_interval,
        'PR_segment': PR_segment,
        'QRS_duration': QRS_duration,
        'QT_interval': QT_interval,
        'ST_segment': ST_segment
    }

def st_slope(signal_df: pd.DataFrame, s_peak: int, t_onset: int) -> float:
    return (signal_df.iloc[t_onset]['ECG_Clean'] - signal_df.iloc[s_peak]['ECG_Clean']) / (t_onset - s_peak)

def extract_ecg_features(record_id: str, channel: int, train_data_folder: Path) -> tuple[pd.DataFrame, list]:
    # Extract the record file
    record = f"{str(train_data_folder.absolute())}{os.sep}{record_id}"
    # Load the header and signals
    header = load_header(record)
    signals, fields = load_signals(record)
    # Get the sampling frequency
    frequency = get_sampling_frequency(header)

    try:
        ecg_signals, info = nk.ecg_process(signals[:, channel], sampling_rate=frequency)
    except:
        return record_id

    hrv_features = nk.hrv_time(ecg_signals, sampling_rate=frequency)

    # Check the intervals (enforce the correct order)
    correct_waves = check_interval(info)

    # If no correct waves are found, return None
    if correct_waves.shape[0] == 0:
        return record_id

    # Compute the ECG features
    correct_waves[["P_wave_duration","PR_interval","PR_segment","QRS_duration","QT_interval","ST_segment"]] = correct_waves.apply(lambda x: ecg_signal_features(x, frequency), axis=1, result_type='expand')
    # Compute the ST slope
    correct_waves['ST_slope'] = correct_waves.apply(lambda x: st_slope(ecg_signals, int(x['ECG_S_Peaks']), int(x['ECG_T_Onsets'])), axis=1)
    # Drop the unnecessary columns
    correct_waves.drop(columns=['ECG_P_Peaks','ECG_P_Onsets','ECG_P_Offsets','ECG_Q_Peaks','ECG_S_Peaks','ECG_T_Peaks','ECG_T_Onsets','ECG_T_Offsets','ECG_R_Peaks'], inplace=True)
    # Add the exam_id
    correct_waves['exam_id'] = record_id
    # Add the heart rate variability features
    correct_waves = correct_waves.groupby('exam_id').aggregate(['mean', 'std', 'min', 'max']).reset_index()
    correct_waves.columns = ['_'.join(col).strip() for col in correct_waves.columns.values]
    correct_waves.rename(columns={'exam_id_': 'exam_id'}, inplace=True)

    return pd.concat([correct_waves, hrv_features], axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--exams', type=str, help='Path to the exams csv file')
    parser.add_argument('--labels', type=str, help='Path to the labels csv file')
    parser.add_argument('--train', type=str, help='Train folder path in WFDB format')
    parser.add_argument('--output', type=str, help='Output folder path to store the extracted features and errors')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--jobs', type=int, default=-1, help='Number of jobs to run in parallel')
    
    args = parser.parse_args()

    exams = pd.read_csv(args.exams)
    labels = pd.read_csv(args.labels)

    train_data_folder = Path(args.train)
    train_records = find_records(train_data_folder.absolute().__str__())

    df = exams[exams['exam_id'].isin([int(x) for x in train_records])].merge(labels, on='exam_id')

    n_positive = df[df['chagas'] == True].shape[0]
    balanced_df = pd.concat([df[df['chagas'] == False].sample(n=n_positive, random_state=args.seed), df[df['chagas'] == True]])
    balanced_records_list = balanced_df.exam_id.values.tolist()

    error_records = []
    channel = 1

    computed_features = Parallel(n_jobs=args.jobs)(delayed(extract_ecg_features)(record_id=record_id, channel=1, train_data_folder=train_data_folder) for record_id in tqdm(balanced_records_list))

    for feature_df in tqdm(computed_features, desc='Computing features'):
        if not isinstance(feature_df, pd.DataFrame):
            error_records.append(feature_df)
            continue

    errors_features = []

    for error_record_id in tqdm(error_records, desc='Computing features for error records'):
        for channel in range(0,12):
            if channel != 1:
                features = extract_ecg_features(error_record_id, channel, train_data_folder=train_data_folder)
                if isinstance(features, pd.DataFrame):
                    errors_features.append(features)
                    break

    all_features = pd.concat([feature for feature in computed_features if isinstance(feature, pd.DataFrame)] + errors_features)

    if not Path(args.output).exists():
        Path(args.output).mkdir(parents=True)

    output_folder = Path(args.output)

    df = all_features.merge(balanced_df[['exam_id','age','is_male','chagas']], on='exam_id', how='inner')

    df = df.drop(columns=['HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2','HRV_SDANN5','HRV_SDNNI5'])

    df.to_csv(output_folder / 'signals_features.csv', index=False)
    error_df = pd.DataFrame(error_records)
    error_df.to_csv(output_folder / 'error_records.csv', index=False)