import numpy as np
import os
import openai
import json
from datetime import datetime, timezone, timedelta
from src.utils import base64_frames_generator
import src.DataAnalysisAgent_LogSampling as ls

# Initialize the OpenAI API client
openai.api_key = 'xxxx'  # Replace with your actual API key


# Save conversation history to a local JSON file
def save_conversation_to_file(conversation_history, file_path="conversation_history.json"):
    with open(file_path, "w") as file:
        json.dump(conversation_history, file, indent=4)


def get_time_series_agent_message():
    # Simulate receiving a message from the time-series analysis agent
    # In practice, this would come from the actual time-series analysis agent
    # Example message format:
    message = {
        "top_hypothesis_pods": [
            {"pod_name": "scenario10-bot-deployment-69bbf69f44-ccqk2_messages",
             "time_interval": "2024-01-14T07:00:00.000Z/2024-01-14T08:00:00.000Z"},
            {"pod_name": "scenario10-malware-deployment-57db4df9f4-p8h7h_messages",
             "time_interval": "2024-01-13T06:00:00.000Z/2024-01-13T07:00:0.000Z"},
            {"pod_name": "ratings-56c584c8d8-rxqlf_messages",
             "time_interval": "2024-01-13T08:00:00.000Z/2024-01-13T09:00:00.000Z"},
            # ... up to 10 pods
        ]
    }
    return message


def get_pod_data_for_hours(pod_name, start_time_dt):
    """
    Get the change_rate and sequence data for the pod for the previous hour, current hour, and next hour.
    If data for any hour does not exist, use the data that is available.
    """
    pod_data = {'change_rate': [], 'sequence': []}
    data_found = 0
    hours_to_get = []

    # Get the previous hour, current hour, and next hour
    prev_hour_dt = start_time_dt - timedelta(hours=1)
    next_hour_dt = start_time_dt + timedelta(hours=1)

    hours_to_get = [prev_hour_dt, start_time_dt, next_hour_dt]

    # Try to load data for each hour
    for hour_dt in hours_to_get:
        date_str = hour_dt.strftime("%m-%d")
        hour_str = hour_dt.strftime("%H")
        # Construct the filenames
        change_rate_filename = f"{date_str}-{hour_str}_change_rate.npy"
        sequence_filename = f"{date_str}-{hour_str}_sequence.npy"
        # Construct the file paths
        change_rate_file_path = os.path.join('time-sequence-matrix', change_rate_filename)
        sequence_file_path = os.path.join('time-sequence-matrix', sequence_filename)
        # Load the data
        if not os.path.exists(change_rate_file_path) or not os.path.exists(sequence_file_path):
            print(f"Data files for {pod_name} at {hour_dt} do not exist.")
            continue  # Skip if data does not exist
        # Load the data
        change_rate_data = np.load(change_rate_file_path, allow_pickle=True)
        sequence_data = np.load(sequence_file_path, allow_pickle=True)
        # Find the pod data in the arrays
        change_rate_for_pod = None
        sequence_for_pod = None

        # Since the data is in the format (257, 2), we can iterate over the array
        for row in change_rate_data:
            if row[0] == pod_name:
                change_rate_for_pod = row[1]
                break

        for row in sequence_data:
            if row[0] == pod_name:
                sequence_for_pod = row[1]
                break

        if change_rate_for_pod is None or sequence_for_pod is None:
            print(f"Data for {pod_name} not found in files at {hour_dt}.")
            continue  # Skip if pod data not found

        # Append data with timestamp
        timestamp_str = hour_dt.strftime("%Y-%m-%dT%H:%M:%S")
        pod_data['change_rate'].append((timestamp_str, change_rate_for_pod))
        pod_data['sequence'].append((timestamp_str, sequence_for_pod))
        data_found += 1

    if data_found == 0:
        print(f"No data found for {pod_name} around {start_time_dt}")
        return None  # or return empty pod_data

    return pod_data


# RUIWU Update: Changed the function name to "analyze_plots_and_samples"("analyze_plots" as original name)
def analyze_plots_and_samples(time_series_message, pod_data_dir, plots_dir, summary_path='temp_output.txt'):
    # RUIWU Update: The folder that contains the log data(Change it to the real folder path when using)
    # pod_data_dir = '../processed_data/log_data/pod_removed'

    # plots_dir = './processed_data/log_plots'
    # plots_dir = './plots'
    # Extract the top hypothesis pods from the message
    hypothesis_pods = time_series_message.get("top_hypothesis_pods", [])

    # Initialize conversation history
    conversation_history = []

    # Collect assistant responses for potential root cause analysis
    assistant_responses = []

    commu_history = []
    pod_data_dict = {}  # Dictionary to hold pod data

    # System message to set context
    system_message = """You are an advanced data analysis agent specialized in identifying the root causes of system issues using detailed analysis of log data from multiple pods.
    Your role is to evaluate patterns, anomalies, and correlations within the logs to validate or refine hypotheses about potential root causes.
    You are expected to provide insights based on the plots of logs for specific top-k ranked log patterns, and inter-pod relationships. Use data-driven reasoning and logical inference to consolidate evidence and determine the most likely root cause(s). Focus on detecting anomalies, correlations, or patterns that point towards root causes.
    Your analysis must be precise and concise. Do not provide any recommendation so far.
    In some cases, there is actually something wrong (anomalies) with the system, but there is no error message in the log. So a pod without errors may contain anomalies (such as high CPU utilization or repeated login requests or other anomalies). So if you see no error message in some pods, just keep doubtful, you can't ignore it.
    Remember that there is only one pod that contains the real root cause, but you can make multiple guesses.The root cause is the fundamental cause that leads to all the errors and anomalies in all the pods in the system. For example, house 1, house 2, house 3 are on fire, and house 4, house 5 are normal. These 5 houses are in the same community, but house 1, 2, 3 are close to each other while house 4, 5 are far away. House 2 is the earliest to be burning, so the root cause pod should be house 2, not house 1, 3, 4 or 5. As it's possible that the fire spread from house 2 to house 1 or 3. And house 4, 5 are far away, so they have nothing to do with this burning issue. The root cause should be something caused the fire, such as a deteriorated cable or a burning cigarette butt.
    Keep in mind that each pod are of the same importance, even if there is little information or few lines of messages in a pod. Some pods with thousands of potential anomaly messages doesn't mean that it must be the pod containing root cause.
    """
    commu_history.append({"role": "system", "content": system_message})

    pod_names_list = []
    for pod_info in hypothesis_pods:
        pod_name = pod_info["pod_name"]
        time_interval = pod_info["time_interval"]
        pod_names_list.append(pod_name)

        # Separate the start time and ending time in time_interval
        start_time_str, end_time_str = time_interval.split('/')
        start_time_dt = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")

        # Get pod data for the hours
        pod_data = get_pod_data_for_hours(pod_name, start_time_dt)
        """if pod_data is None:
            continue  # Skip if not enough data"""

        # Store the data in the dictionary
        if pod_data:
            pod_data_dict[pod_name] = {
                'time_interval': time_interval,
                'change_rate': pod_data['change_rate'],
                'sequence': pod_data['sequence']
            }
        else:
            pod_data_dict[pod_name] = {
                'time_interval': time_interval,
                'change_rate': "Pod information not enough or pod file not found",
                'sequence': "Pod information not enough or pod file not found"
            }

        # Generate base64-encoded image (assuming you have a function for this)
        # encoded_image = get_base64_encoded_image(pod_name, plots_dir)

        # RUIWU Update: Separate the start time and ending time in time_interval
        start_time, end_time = ls.message_catch(time_interval)
        print(start_time)
        print(end_time)

        plot_filename = pod_name + ".png"

        try:
            plot_file_path = os.path.join(plots_dir, plot_filename)
        except Exception as e:
            print(f"Fail to read plot: {e}")
            plot_file_path = None
        """if not os.path.exists(plot_file_path):
            print(f"Plot file {plot_file_path} does not exist.")
            continue"""

        # RUIWU Update: Get the whole pod data
        pod_content = ls.find_pod(pod_name, pod_data_dir)
        # RUIWU Update: Get the sampled pod data
        sampled_data, selected_time = ls.pod_sampling(start_time, end_time, pod_content, sample_size=10)

        # Generate base64-encoded image
        try:
            base64_frames = list(base64_frames_generator([plot_file_path]))
        except Exception as e:
            print(f"Fail to read plot: {e}")
            base64_frames = None
        """if not base64_frames:
            print(f"Failed to encode image {plot_file_path}.")
            continue"""
        if (base64_frames):
            encoded_image = base64_frames[0]
        else:
            encoded_image = None

        # Include the pod data in the final message
        try:
            pod_data_text = "\n\n".join(
                [
                    f"Pod: {pod_name}\nTime Interval: {data['time_interval']}\nChange Rate: {data['change_rate']}\nSequence: {data['sequence']}"
                    for pod_name, data in pod_data_dict.items()]
            )
        except Exception as e:
            print(f"Fail to get pod_data_text: {e}")
            pod_data_text = None

        # User message, including the time interval to focus on
        if pod_data is None or plot_file_path is None or not os.path.exists(
                plot_file_path) or not base64_frames or pod_data_text is None or encoded_image is None:
            messages = [{"role": "user",
                         "content": f"Here are 10 pieces of randomly sampled messages from pod {pod_name} during the period {time_interval}: {sampled_data}.As for your response of this message, it must be within 300 words. Please provide a structured and concise summary of the pod's behavior of the time interval."}]
        else:
            messages = [{
                "role": "user",
                "content": [
                    f"Here is the log plot of the pod: {pod_name}. Please analyze the time interval: {time_interval}.",
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{x}",
                            },
                        },
                        [encoded_image]),
                    # RUIWU Update: Also put the sampled log data into the prompt
                    f"Here are 10 pieces of randomly sampled log messages from pod {pod_name} during the period {time_interval}: {sampled_data}",
                    f"Here is the time-series data for this pod over one a three-hour interval which inclues hour before to one hour after the original time interval {time_interval}. The analysis should focus on sudden spikes or changes in time series for the metrics data:\n\n",
                    f"{pod_data_text}\n\n",
                    "As for your response of this message, it must be within 220 words. Please provide a structured and concise summary of the pod's behavior of the time interval. "
                    ],
            }]
        # print(commu_history)
        # API call to analyze the plot
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Replace with the appropriate model
            messages=messages,
            max_tokens=500
        )

        # Extract and save the assistant's reply
        assistant_message = response.choices[0].message.content
        commu_history.append({"role": "assistant", "content": assistant_message})
        conversation_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pod_name": pod_name,
            "time_interval": time_interval,
            "assistant_response": assistant_message
        }
        conversation_history.append(conversation_entry)

        assistant_responses.append(assistant_message)

        # Output the assistant's reply
        print(f"Analysis of {pod_name} during {time_interval}:\n{assistant_message}\n")

    with open(summary_path, 'r', encoding='latin-1') as file:
        data_content_str = file.read()
        file.close()

    step2_prompt = f"""
    Now, I've also got the summary of log files for your reference:{data_content_str}. This summary was extracted from a csv file, so you can read the summary as if it's a csv file. You can refer to the summary and the information that I gave you previously as resources for you to perform your role. 
    Pay attention to the earliest error time and earliest error message, these information may provide you with some useful clues to find the root cause and root cause pod.
    After reading the summary, you can make a guess about the relationships between the pods. For example, pod i->pod j, pod j->pod k, pod k-> pod r, pod k->pod i, pod i not->pod e (you replace pod i, pod k, pod j, pod r, pod e with real pod names. '->' represents related to, and it's also the route that the anomalies is passed through just like the spread of fire. 'not ->' represents not related to, and anomalies doesn't go through this relationship.). For example, in the burning house case that I gave you, the relationships should be house 2->house 1; house 1->house 3; house 2-> house3; house 2 not-> house 4; house 2 not-> house 5.
    Please give me your guess about the relationship as detailed as possible, and you can give me some very brief explanations for the reason of your guess. You have a maximum of 1600 words for this part. You can only make 4 guesses, each guess should contain the relationships between all the pods in pod names shown in the summary of log files.
    After that, give me the name of 20 pods that you think should be focused on. The format is like '20 pods that should be focused on: ...'.
    """

    commu_history.append({"role": "user", "content": step2_prompt})

    response = openai.chat.completions.create(
        model="gpt-4o",  # Replace with the appropriate model
        messages=commu_history,
        max_tokens=500
    )
    assistant_message = response.choices[0].message.content

    commu_history.append({"role": "assistant", "content": assistant_message})

    print("Data Analysis Agent: Combining pod sample with summary info")

    step3_prompt = """
    Now, you have 20 chances to get some samples of raw data in the log files of pods. You respond to me with the format 'pod name: the name of pod that you are interested in; start time: the start time of period that you are interested in; end time: the end time of the period that you are interested in'. I name this format as 'format 1'. If you deliver your message with this format, I'll give you some samples in the corresponding pod and time period.
    For example, your respond can be 'scenario10-malware-deployment-57db4df9f4-p8h7h; 2024-01-13,06:45; 2024-01-13,09:45'. Please follow this format strictly, especially for the time format, no matter what you are taught in the previous prompts. You can only request samples for only one pod and only one time period in a response. There is no limit for the time period, you can select any period of time such as 1 hour, 2 hours, 3 hours, 4 hours, you name it. The start time and end time are also not restricted, you select the start time and end time that you are interested in.
    You should keep asking for data until I deliver 'STOP' to you. You should ask for data from the pod that you think is the most likely to be the pod with root cause. I suggest you ask for data from different pods at each time, but you can ask for data from the same pod for at most 2 times.
    You can only deliver your message in format 1 before you receive 'STOP'. Don't deliver extra message when you are delivering in format 1. Don't deliver in formats other than format 1 before you receive 'STOP'. You can't deliver 'STOP' to me and you can't stop asking me for data until I deliver 'STOP' to you. But after I deliver 'STOP' to you (you receive 'STOP'), you should summarize your findings from the pod data within 500 words and you can also deliver in other formats after I deliver 'STOP' to you.
    """

    commu_history.append({"role": "user", "content": step3_prompt})

    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Replace with the appropriate model
        messages=commu_history,
        max_tokens=500
    )

    assistant_message = response.choices[0].message.content

    for i in range(21):
        if i != 20:
            try:
                commu_history.append({"role": "assistant", "content": assistant_message})
                print(assistant_message)
                pod_name, start_time, end_time = ls.agent_step3_message_get(assistant_message)
                print(f"Start time:{start_time}, End time:{end_time}")
                pod_content = ls.find_pod(pod_name, pod_data_dir)
                sampled_data, selected_time = ls.pod_sampling(start_time, end_time, pod_content, sample_size=10)
                pod_samples_prompt = f"Here are some samples for pod {pod_name} during the period {start_time}: {end_time}. Samples: {sampled_data}"
                commu_history.append({"role": "user", "content": pod_samples_prompt})

                response = openai.chat.completions.create(
                    model="gpt-4o-mini",  # Replace with the appropriate model
                    messages=commu_history,
                    max_tokens=500
                )
                assistant_message = response.choices[0].message.content
                commu_history.append({"role": "assistant", "content": assistant_message})
            except Exception as e:
                print(f"Format unconsistant!, error info:{e}")
        else:
            commu_history.append({"role": "user", "content": 'STOP'})
            response = openai.chat.completions.create(
                model="gpt-4o",  # Replace with the appropriate model
                messages=commu_history,
                max_tokens=500
            )
            assistant_message = response.choices[0].message.content
            commu_history.append({"role": "assistant", "content": assistant_message})
    pod_names_str = ','.join(pod_names_list)
    final_message = f"""
    Now, you need to summarize and analyze root cause based on your previous analysis.
    Your response should be:
    -Provide a ranked Top-10 list of the most likely root cause pods, along with the root cause.
    -Summarize key insights or patterns from the analysis of the provided log data.
    -Offer clear and actionable recommendations for resolving or mitigating the identified issues.
    Please note that the root cause pods in your response should only be chosen from this list: {pod_names_str}.
    You can review your guess for relationships between pods, the summary of log files and all existing information to figure out the root cause and root cause pod.
    """
    commu_history.append({"role": "user", "content": final_message})
    print(f"Pod names to choose from: {pod_names_str}")

    final_response = openai.chat.completions.create(
        model="gpt-4o",  # Replace with the appropriate model
        messages=commu_history,
        max_tokens=1000,
    )

    final_assistant_message = final_response.choices[0].message.content

    commu_history.append({"role": "assistant", "content": final_assistant_message})

    # Output the final report
    print("Final Root Cause Analysis Report:\n")
    print(final_assistant_message)

    # Save the final report in the conversation history
    conversation_history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "assistant_response": final_assistant_message
    })

    # Save conversation history to a local file
    save_conversation_to_file(conversation_history)


if __name__ == "__main__":
    # Get the message from the time-series analysis agent
    time_series_message = get_time_series_agent_message()

    # Analyze the plots based on the message

    analyze_plots_and_samples(time_series_message, pod_data_dir='path_to_pod_data', plots_dir='path_to_plots',
                              summary_path='temp_output.txt')

