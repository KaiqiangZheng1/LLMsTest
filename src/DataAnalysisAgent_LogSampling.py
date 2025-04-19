import os
import re
from datetime import datetime
import random


def message_catch(message):
    # Split the time_interval into start and end times
    start_str, end_str = message.split('/')

    # Parse the start and end datetime strings
    start_dt = datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    end_dt = datetime.strptime(end_str, '%Y-%m-%dT%H:%M:%S.%fZ')

    # Format the datetime objects to the desired format
    start_time = start_dt.strftime('%Y-%m-%d,%H:%M')
    end_time = end_dt.strftime('%Y-%m-%d,%H:%M')

    # Output the results
    return start_time, end_time


# folder_path = r"C:\Users\jason\Desktop\ECE1786\Project\20240115\log_data\pod_removed"
def find_pod(pod_name, folder_path):
    content = None

    # Traverse the folder and find the file with the selected pod_name
    for file_name in os.listdir(folder_path):
        if pod_name in file_name:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                break  # If pod file found, end the loop
            except Exception as e:
                # If fail to read the file, return None
                print(f"Pod found but fail to read file: {e}")
                return None

    if content:
        return content
    else:
        print(f"No file found containing '{pod_name}' in its name.")
        return None


# out_of_interest: the timestamps of data that has already been fed into the agent
# start_time/end_time: date,time(e.g. 2024-11-22,15:34)
def pod_sampling(start_time, end_time, pod_data, sample_size = 5, out_of_interest=None):
    from datetime import datetime
    import random

    def truncate_to_seconds(dt):
        return dt.replace(microsecond=0)

    # Convert start_time and end_time to datetime objects
    start_time_dt = datetime.strptime(start_time, '%Y-%m-%d,%H:%M')
    end_time_dt = datetime.strptime(end_time, '%Y-%m-%d,%H:%M')

    # Parse out_of_interest timestamps as datetime objects truncated to seconds
    if (out_of_interest != None):
        out_of_interest_timestamps = set()
        for line in out_of_interest.strip().split('\n'):
            line = line.strip()
            if line:
                # Parse timestamp
                try:
                    timestamp = datetime.strptime(line, '%Y-%m-%dT%H:%M:%S.%fZ')
                except ValueError:
                    # Handle case where milliseconds are missing
                    try:
                        timestamp = datetime.strptime(line, '%Y-%m-%dT%H:%M:%SZ')
                    except ValueError:
                        continue  # Skip invalid format

                # Truncate timestamp to seconds
                timestamp_truncated = truncate_to_seconds(timestamp)
                out_of_interest_timestamps.add(timestamp_truncated)
    else:
        out_of_interest_timestamps = None

    # Parse pod_data lines and extract timestamp and line
    pod_data_lines = []
    if pod_data:
        for line in pod_data.strip().split('\n'):
            line = line.strip()
            if line:
                # Extract the timestamp at the beginning of the line
                parts = line.split(None, 1)  # Split on whitespace, maxsplit=1
                if len(parts) >= 1:
                    timestamp_str_raw = parts[0]
                    try:
                        timestamp = datetime.strptime(timestamp_str_raw, '%Y-%m-%dT%H:%M:%S.%fZ')
                    except ValueError:
                        # Handle case where milliseconds are missing
                        try:
                            timestamp = datetime.strptime(timestamp_str_raw, '%Y-%m-%dT%H:%M:%SZ')
                        except ValueError:
                            continue  # Skip lines with invalid timestamp format

                    pod_data_lines.append((timestamp, timestamp_str_raw, line))

    # Sort pod_data_lines by timestamp
    pod_data_lines.sort(key=lambda x: x[0])

    # Filter data between start_time and end_time, excluding out_of_interest
    filtered_data = []
    for timestamp, timestamp_str, line in pod_data_lines:
        timestamp_truncated = truncate_to_seconds(timestamp)
        if out_of_interest != None:
            if start_time_dt <= timestamp <= end_time_dt and timestamp_truncated not in out_of_interest_timestamps:
                filtered_data.append((timestamp, timestamp_str, line))
        else:
            if start_time_dt <= timestamp <= end_time_dt:
                filtered_data.append((timestamp, timestamp_str, line))

    # Randomly sample 10 lines
    if len(filtered_data) >= sample_size:
        sampled_data_lines = random.sample(filtered_data, sample_size)
    else:
        sampled_data_lines = filtered_data  # Less than 10 lines available

    # Extract sampled data and their timestamps
    sampled_data = [line for timestamp, timestamp_str, line in sampled_data_lines]
    selected_timestamp = [timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ') for timestamp, timestamp_str, line in
                          sampled_data_lines]

    return sampled_data, selected_timestamp


def agent_step3_message_get(message):
    if message.strip() == 'STOP':
        return 'STOP'
    else:
        parts = message.split(';')
        parts = [part.strip() for part in parts]

        if len(parts) != 3:
            # 如果格式不正确，返回 None
            return None

        pod_name, start_time, end_time = parts

        # 移除可能存在的前缀
        if 'pod name:' in pod_name.lower():
            pod_name = pod_name.split(':', 1)[1].strip()
        if 'start time:' in start_time.lower():
            start_time = start_time.split(':', 1)[1].strip()
        if 'end time:' in end_time.lower():
            end_time = end_time.split(':', 1)[1].strip()

        return pod_name, start_time, end_time
