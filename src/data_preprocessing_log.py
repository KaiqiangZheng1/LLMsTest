import os
import pandas as pd
import re
from collections import Counter
import datetime
from openai import OpenAI


def file_catalog(file_path):
    # filePath = 'C:\\Users\\jason\\Desktop\\ECE1786\\Project\\20240115\\log_data\\pod_removed\\'
    # Get the list of file name
    file_list = os.listdir(file_path)
    for i, filename in enumerate(file_list):
        file_list[i] = file_path + file_list[i]

    # Create a dataframe which contains the file name
    df = pd.DataFrame(file_list, columns=['filename'])
    df = df.dropna()
    return df


# This function checks if there is error info in the log file
def check_file_for_errors(full_path):
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if 'stderr' in content or 'error' in content or 'fail' in content or 'bad' in content:
                return 'error'
            else:
                return 'normal'
    except Exception as e:
        # If file unable to be opened, return a message
        print(f"Fail to read file: {e}")
        return 'file not found or unreadable'


# This function counts the number of chars in a file
def count_file_characters(full_path):
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
            num_chars = len(content)
            return num_chars
    except Exception as e:
        # If fail to read the file, return None
        print(f"Fail to read file: {e}")
        return None


# This function check if the file can be directly fed into the LLM agent
def set_directly_feedable_flag(char_count):
    if char_count is None:
        return 'file not found or unreadable'
    if char_count > 1000:
        return 'not directly feedable'
    if char_count <= 1000 and char_count != None:
        return 'directly feedable'


# This function sums the number of characters that are in all directly feedable files, so that we know how many tokens are going to be used
def char_total_sum(df):
    filtered_df = df[(df['char_count'].notna()) & (df['directly feedable flag'] == 'directly feedable')]
    char_total_sum = filtered_df['char_count'].sum()
    return char_total_sum


# This function calculates the repetition rate of contents in a file, so that we know something about the variance of information in a file
def calculate_repetition_rate(full_path):
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Split words
            words = re.findall(r'\b\w+\b', text.lower())
            total_words = len(words)
            word_counts = Counter(words)
            # Calculate repeated words
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            # Calculate repetition rate
            repetition_rate = repeated_words / total_words if total_words else 0
            return repetition_rate
    except Exception as e:
        print(f"Fail to read file: {e}")
        return None


# This function adds the repetition rate as a column in an existing pandas dataframe
def repetition_rate_to_df(df):
    repeatition_rate = []
    for file_path in df['filename']:
        rate = calculate_repetition_rate(file_path)
        repeatition_rate.append(rate)
    df['repeatition'] = repeatition_rate
    df = df.dropna()
    return df


def find_earliest_error(full_path):
    search_terms = ['stderr', 'error', 'fail', 'bad']

    # Initialize parameters
    earliest_timestamp = None  # Earliest time stamp（datetime object）
    earliest_timestamp_str = None  # Earliest time stamp string variable
    earliest_lines = []  # Earliest error info

    with open(full_path, 'r') as f:
        for line in f:
            # Check if error keywords exists
            if any(term in line for term in search_terms):
                # Extract the timestamp
                timestamp_str = line.split()[0]
                # Parse the timestamp
                timestamp_str_clean = timestamp_str.rstrip('Z')
                try:
                    timestamp = datetime.datetime.strptime(timestamp_str_clean, '%Y-%m-%dT%H:%M:%S.%f')
                except ValueError:
                    # If no microsecond component in timestamp, use another pattern to parse
                    timestamp = datetime.datetime.strptime(timestamp_str_clean, '%Y-%m-%dT%H:%M:%S')
                # Compare and get the earliest timestamp along with the error information
                if earliest_timestamp is None or timestamp < earliest_timestamp:
                    earliest_timestamp = timestamp
                    earliest_timestamp_str = timestamp_str
                    line = re.sub(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z', '', line)
                    line = re.sub(r'[\t\n\\]', '', line)
                    earliest_lines = [line]
                elif timestamp == earliest_timestamp:
                    earliest_lines.append(line)
        return earliest_lines, earliest_timestamp_str


def extract_pod_names(df):
    import re
    pattern = r'([^\\]+)_messages$'

    def extract_name(x):
        match = re.search(pattern, x)
        if match:
            return match.group(1)
        else:
            return x

    df['pod name'] = df['filename'].apply(extract_name)
    return df

def remove_unwanted_substring(df, folderpath):
    unwanted_string = folderpath
    df['pod name'] = df['pod name'].str.replace(unwanted_string, '', regex=False)
    df['pod name'] = df['pod name'].str.replace('.txt', '', regex=False)
    df['pod name'] = df['pod name'].str.replace('.csv', '', regex=False)
    return df


# This function parses all the files with all needed functions and returns the ultimately needed dataframe
# 'C:\\Users\\jason\\Desktop\\ECE1786\\Project\\20240115\\log_data\\pod_removed\\'
def all_file_parse(folder_path):
    df = file_catalog(folder_path)
    # Apply the error check function to all files
    df['error or normal'] = df['filename'].apply(check_file_for_errors)
    # Count number of characters in all files
    df['char_count'] = df['filename'].apply(count_file_characters)
    df = df.dropna()
    # Check if the file can be directly fed into the LLM agent
    df['directly feedable flag'] = df['char_count'].apply(set_directly_feedable_flag)
    # Calculate reprtition rate and store it in a new column in df
    # df = repetition_rate_to_df(df)
    # Get the earliest error timestamp along with error info
    earliest_error_messages = []
    earliest_time_stamps = []
    for datafile in df['filename']:
        err_message, ear_timestamp = find_earliest_error(datafile)
        earliest_error_messages.append(err_message)
        earliest_time_stamps.append(ear_timestamp)
    df['earliest error content'] = earliest_error_messages
    df['earlist error time'] = earliest_time_stamps
    df = extract_pod_names(df)
    df = remove_unwanted_substring(df, folder_path)
    return df


def get_small_files(df):
    # Filter small files
    df_feedable = df[df['directly feedable flag'] == 'directly feedable']
    # Save file name to df_feedable['file_feedable']
    df_feedable = df_feedable[['filename']].rename(columns={'filename': 'file_feedable'})
    return df_feedable


# This function takes in the ultimately needed dataframe and returns the prompt (in the string format) which can be used in other files
def get_log_prompt(df):
    needed_columns = ['error or normal', 'earliest error content', 'earlist error time', 'pod name']
    new_df = df[needed_columns].copy()
    try:
        new_df.to_csv("temp_output.txt", index=False)
    except PermissionError:
        print("Permission denied: Unable to write to the file.")
    except Exception as e:
        print(f"An error occurred: {e}")
    with open('temp_output.txt', 'r', encoding='latin-1') as file:
        data_content_str = file.read()
        file.close()
    sys_role = """You are an expert on-call system engineer, your role is to analyze the processed log file about pods in a micro service system. There is only one pod that is the root cause pod. You need to give me your hypotheses which are 10 lines about 10 possible pods that might be the root cause pod. Your hypotheses should be in the following format:
-Hypothesis: 1, root cause pod: pod name, root cause node: event.
-Hypothesis: 2, root cause pod: pod name, root cause node: event.
…
-Hypothesis: 10, root cause pod: pod name, root cause node: event.
The ‘pod name’ should be a string within the column named ‘pod name’ in the data that I provide you. The root cause node is a fundamental event that directly caused all errors in the system. Therefore, the ‘event’ argument should be your own hypotheses of the root cause node which is less than 30 words long each. There is only one root cause event in each hypothesis.
"""
    hint_str = f"""The log data is composed of {len(new_df.columns)} columns.
    The column {new_df.columns[0]} contains the information which indicates if the log file contains error messages. ‘normal’ means that there isn't any error messages in the file while ‘error’ means that there is one or more error messages in the file.
    The column {new_df.columns[1]} contains the earliest error message that appears in the file.
    The column {new_df.columns[2]} contains the earliest time that the error occurred in the file, this time is shown in a format of a timestamp. You can figure out the date and time from the timestamp.
    The column {new_df.columns[3]} contains the pod name, each pod name represents a pod in the system.
    The following is the summary of the log data: {data_content_str}. The content of small log files are also provided for your reference: """
    small_file_df = get_small_files(df)
    small_file_df.to_csv("small_file_df.csv")
    small_file_content = ""
    for feedable_file in small_file_df['file_feedable']:
        with open(feedable_file, 'r', encoding='latin-1') as myfile:
            mycontent = myfile.read()
            temp_str = f"file name {feedable_file}, file content {mycontent}."
            small_file_content = small_file_content + temp_str
            file.close()
    full_prompt = hint_str + small_file_content
    return sys_role, full_prompt

#For test use only
def LLM_communicate(system_role, full_prompt, private_api_key):
    client = OpenAI(api_key=private_api_key)
    agent_hist = []
    agent_hist.append({"role": "system", "content": system_role})
    agent_hist.append({"role": "user", "content": full_prompt})
    first_step = client.chat.completions.create(
        model="gpt-4o",
        messages=agent_hist
    )
    agent_hist.append({"role": "assistant", "content": first_step.choices[0].message.content})
    return first_step.choices[0].message.content
