import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
# import base64

pattern_change_rate = re.compile(r'pictures[\\/]+pod_change_rate_batch(\d+)\.png', re.IGNORECASE)
pattern_receive_bandwidth = re.compile(r'pictures[\\/]+pod_receive_bandwidth_batch(\d+)\.png', re.IGNORECASE)
from src.utils import base64_frames_generator
'''def base64_frames_generator(file_paths):
    print(1)
    for file_path in file_paths:
        print(2)
        with open(file_path, 'rb') as f:
            image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            yield encoded_image'''

def frame_list_generator(saved_file_paths):
    change_rate_files = []
    receive_bandwidth_files = []
    for f in saved_file_paths:
        match_cr = pattern_change_rate.search(f)
        match_rb = pattern_receive_bandwidth.search(f)
        if match_cr:
            batch_number = int(match_cr.group(1))
            change_rate_files.append((batch_number, f))
        if match_rb:
            batch_number = int(match_rb.group(1))
            receive_bandwidth_files.append((batch_number, f))

    change_rate_files.sort(key=lambda x: x[0])
    receive_bandwidth_files.sort(key=lambda x: x[0])

    batch_numbers = sorted(set([x[0] for x in change_rate_files]) & set([x[0] for x in receive_bandwidth_files]))

    frames_list = []
    batch_size = 1
    for i in range(0, len(batch_numbers), batch_size):
        batch_nums = batch_numbers[i:i + batch_size]

        batch_file_paths = []
        for num in batch_nums:
            # 获取对应的 change_rate 和 receive_bandwidth 文件路径
            cr_file = next((x[1] for x in change_rate_files if x[0] == num), None)
            rb_file = next((x[1] for x in receive_bandwidth_files if x[0] == num), None)
            if cr_file and rb_file:
                batch_file_paths.extend([cr_file, rb_file])

        # 生成迭代器
        frame_iterator = base64_frames_generator(batch_file_paths)
        frames_list.append(frame_iterator)

    return frames_list

def metrics_data_preprocess(file_path, processed=True):
    i = 0
    pictures_folder = 'processed_data/metrics_pictures'
    time_sequence_folder = 'time-sequence-matrix'

    if not os.path.exists(pictures_folder):
        os.makedirs(pictures_folder)
    if not os.path.exists(time_sequence_folder):
        os.makedirs(time_sequence_folder)

    if processed:
        saved_file_paths = [os.path.join(pictures_folder, f) for f in os.listdir(pictures_folder) if f.endswith('.png')]
        frames_list = frame_list_generator(saved_file_paths)
        return frames_list

    data_pod_transmit = np.load(file_path, allow_pickle=True).item()
    data_pod_transmit = pd.DataFrame(data_pod_transmit)

    time_values = pd.to_datetime([int(t) for t in data_pod_transmit.loc['time', 'scenario8_app_request']], unit='s')
    print('Length of time_values:', len(time_values))

    pod_names = data_pod_transmit.loc['Pod_Name'][0]
    print('Length of pod_names:', len(pod_names))

    sequences_data = np.array(data_pod_transmit.loc['Sequence'][0])
    print('sequences_data shape:', sequences_data.shape)

    num_time_points, num_columns = sequences_data.shape
    num_pod_names = len(pod_names)

    if num_columns != num_pod_names:
        print('Mismatch detected between number of columns in sequences_data and length of pod_names.')
        print('First few values of the first column in sequences_data:')
        print(sequences_data[:5, 0])
        sequences_data = sequences_data[:, :-1]
        num_columns -= 1
        print('Removed the last column from sequences_data.')
        print('New sequences_data shape:', sequences_data.shape)

    assert sequences_data.shape[1] == len(pod_names), \
        "After adjustment, the number of columns in sequences_data and length of pod_names still do not match."

    sequences = pd.DataFrame(sequences_data, columns=pod_names, index=time_values)
    print('Created sequences DataFrame with shape:', sequences.shape)

    sequences_5min = sequences.resample('5min').sum()

    prev_values = sequences_5min.shift(1).replace(0, 1)
    change_rate = (sequences_5min - prev_values) / prev_values
    change_rate.replace([np.inf, -np.inf], np.nan, inplace=True)
    change_rate.fillna(0, inplace=True)

    pod_names = sequences.columns.tolist()
    num_pods = len(pod_names)
    batch_size = 10
    n_cols = 5

    saved_file_paths = []

    for batch_num, batch_start in enumerate(range(0, num_pods, batch_size)):
        batch_pod_names = pod_names[batch_start: batch_start + batch_size]
        n_pods_in_batch = len(batch_pod_names)
        n_rows = math.ceil(n_pods_in_batch / n_cols)

        fig1, axs1 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axs1 = axs1.flatten()

        for idx, pod_name in enumerate(batch_pod_names):
            ax = axs1[idx]
            sequence_values = sequences_5min[pod_name]
            ax.plot(sequence_values.index, sequence_values.values)
            ax.set_title(pod_name, fontsize='small')

            if idx % n_cols == 0:
                ax.set_ylabel('Sum Sequence Value')
            else:
                ax.set_yticklabels([])

            if idx >= (n_rows - 1) * n_cols:
                ax.set_xlabel('Time')
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
            else:
                ax.set_xticklabels([])

        for idx in range(n_pods_in_batch, n_rows * n_cols):
            fig1.delaxes(axs1[idx])

        fig1.tight_layout()
        output_file1 = os.path.join(pictures_folder, f'pod_receive_bandwidth_batch{batch_num + 1}.png')
        fig1.savefig(output_file1)
        plt.close(fig1)
        saved_file_paths.append(output_file1)
        print(f"Plot saved as {output_file1}")

        fig2, axs2 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axs2 = axs2.flatten()

        for idx, pod_name in enumerate(batch_pod_names):
            ax = axs2[idx]
            change_rate_values = change_rate[pod_name]
            ax.plot(change_rate_values.index, change_rate_values.values)
            ax.set_title(pod_name, fontsize='small')

            if idx % n_cols == 0:
                ax.set_ylabel('Change Rate')
            else:
                ax.set_yticklabels([])

            if idx >= (n_rows - 1) * n_cols:
                ax.set_xlabel('Time')
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
            else:
                ax.set_xticklabels([])

        for idx in range(n_pods_in_batch, n_rows * n_cols):
            fig2.delaxes(axs2[idx])

        fig2.tight_layout()
        output_file2 = os.path.join(pictures_folder, f'pod_change_rate_batch{batch_num + 1}.png')
        fig2.savefig(output_file2)
        plt.close(fig2)
        saved_file_paths.append(output_file2)
        print(f"Plot saved as {output_file2}")

    hourly_groups = sequences_5min.groupby(sequences_5min.index.floor('H'))

    for timestamp, group in hourly_groups:

        sequence_hourly_data = []
        for pod_name in group.columns:
            sequence_values = group[pod_name].values
            sequence_hourly_data.append(sequence_values)

        df_sequence_hour = pd.DataFrame({
            'Pod_Name': group.columns,
            'Sequence': sequence_hourly_data
        })

        sequence_filename = timestamp.strftime('%m-%d-%H') + '_sequence.npy'
        sequence_file_path = os.path.join(time_sequence_folder, sequence_filename)
        np.save(sequence_file_path, df_sequence_hour.values)
        print(f"Hourly sequence data saved as {sequence_file_path}")

        change_rate_group = change_rate.loc[group.index]
        change_rate_hourly_data = []
        for pod_name in change_rate_group.columns:
            change_rate_values = change_rate_group[pod_name].values
            change_rate_hourly_data.append(change_rate_values)

        df_change_rate_hour = pd.DataFrame({
            'Pod_Name': change_rate_group.columns,
            'Change_Rate': change_rate_hourly_data
        })

        change_rate_filename = timestamp.strftime('%m-%d-%H') + '_change_rate.npy'
        change_rate_file_path = os.path.join(time_sequence_folder, change_rate_filename)
        np.save(change_rate_file_path, df_change_rate_hour.values)
        print(f"Hourly change rate data saved as {change_rate_file_path}")

    frames_list = frame_list_generator(saved_file_paths)
    return frames_list
