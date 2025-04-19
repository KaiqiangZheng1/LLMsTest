import os
from pathlib import Path
from src.llm_infer import LLM
import base64


class Generation:
    def __init__(self, config, logger):
        self.config = config
        self.llm = LLM(config, logger)
        self.logger = logger

    def generate_metrics_analysis(self, frames):
        system = " You are a data scientist specializing in analyzing pod metrics from images. Each image contains multiple subplots of different pods' metrics over time. Your task is to analyze each small graph within the provided images, focusing on the behavior patterns, trends, and anomalies of each pod."
        instruction = f"""

           Your objectives are as follows:

           - For each pod represented in the images, analyze its behavior over time, noting any patterns, trends, or anomalies.
           - Provide a detailed summary with time annotations for each pod (e.g., "Stable operation from 1-13 6 to 1-13 7, sudden abnormal increase at 1-14 7 AM", etc.).
           - Summarize the findings and associate them with the corresponding pod names.

           Please ensure that your analysis is thorough and precise.

           Your output should be a list of dictionaries, where each dictionary contains:

           - 'pod_name': The name of the pod.
           - 'summary': A detailed summary of the pod's behavior, including time annotations of change points.
           - 'list_of_sudden_change_times': A list timestamp where the trend of the data suddenly changed.
           - 'list_of_sudden_change_behavor': A list sudden change behavior that's corresponding to list_of_sudden_change_times. Describe the proximate magnitude as well.

           If any images are missing or cannot be analyzed, please note this in your response.
           """

        self.logger.info("Hypothesis Generation Inferencing ...")

        return self.llm.llm_generate(
            self.config.llm_model,
            instruction,
            frames,
            system,
            max_tokens=self.config.max_token
        )

    def generate_likely_hypothesis(self, metrics_summary, log_text):
        system = "You are an experienced data scientist. You are very familiar with root cause analysis, which is a task of finding the root casue pod among all abnormal pods in an incident."

        instruction = f"""Based on the metrics_pictures and log_data, conduct a detailed root cause analysis of any issues or anomalies observed by thinking step by step. 
        
        The metrics_summary includes: 1.pod_name, The name of the pod.
                                      2.summary, A detailed summary of the pod's behavior, including time annotations.
        The metric summary is provided as below:{metrics_summary}

        The log_data has been pre-processed to capture a few important information: {log_text}        

        The output is expected to contain these variables and structured in JSON format, respond with this JSON file only:
        1. root_cause_analysis_summerization_result: Consider all the summaries together and explain any relationships or patterns that could indicate the root cause. Include example of the most important and earliest timestamp and log information as well.
        2. all_root_cause_hypothesis: Consider root_cause_analysis_summerization_result and all data available, propose 3 root_cause_hypothesis. Each root_cause_hypothesis contains:
         2.1 incident_time_interval: The starting time of one hour timeframe that includes the starting time of the incident. The format should be 'month-day-hour'. Example: For 2024-01-14T06:00:00Z the output should be 01-14-06.
         2.2 root_cause_pod: The root cause pod of the incident.
         2.3 hypothesis_explanation: An explanation of the hypothesis together with supporting evidence including timestamp.
         2.4 neighboring_pods: A list of 10 pod names with time-series metric pattern in metrics_pictures similar to root_cause_pod.
        """
        
        instruction_direct_to_data = f"""The system that you deployed in AWS Elastic Container Kubernetes service is experiencing an incident.
        Based on the metrics_pictures and log_data generated from this service, conduct a detailed root cause analysis of any issues or anomalies observed by thinking step by step. 
        
        In root cause analysis, the goal is to identify one failure point that's causing all the anomalies happened in later time.
        
        The metrics_summary includes: 1.pod_name, The name of the pod.
                                      2.summary, A detailed summary of the pod's behavior, including time annotations.
        The metric summary is provided as below:{metrics_summary}

        Sudden change in magnitude and spikes in metrics (both sequence and change rate) are more likely to indicate the pod is the root cause pod.

        The log_data has been pre-processed to capture important information: {log_text}  

        Errors in log are more likely to indicate the pod is the root cause pod.

        Consider all the log and metrics data and propose 10 root_cause_hypothesis. Each root_cause_hypothesis contains:
         incident_time_interval: The starting time of one hour timeframe that includes the starting time of the incident.
         hypothesis_explanation: An explanation of the hypothesis together with supporting evidence including timestamp.

        Each root_cause_hypothesis should have only one root cause pod and try to think carefully. Overall, there is only one root cause pod that causes all the errors in the system.
         
        Based on these root_cause_hypothesis and all the metrics and logs data, generate output with containing these variables and structured in JSON format, respond with this JSON file only without comments. Also make sure there's no duplicates in pod_name:
        1. top_hypothesis_pods: Top 10 most likely pods to be the root cause pod that are identified among all the pods. Order by the likelyness with first element being the most likely one.
         1.1 pod_name: The pod name of this current top_hypothesis_pods.
         1.2 time_interval: The time interval of the hypothesis incident_time_interval. Example format: "2024-01-13T08:00:00.000Z/2024-01-13T09:00:00.000Z"
        """

        #1.3 hypothesis_pods_explaination: The explaination of why the pod is identified as root cause pod.

        self.logger.info("Hypothesis Generation Inferencing ...")

        return self.llm.llm_generate_text_input(
            self.config.llm_model,
            instruction_direct_to_data,
            system,
            max_tokens=self.config.max_token,
        )


    def generate_benchmark(self, metrics_summary, log_text):
        system = "You are an experienced data scientist. You are very familiar with root cause analysis, which is a task of finding the root casue pod among all abnormal pods in an incident."

        instruction = f"""Based on the metrics_pictures and log_data, conduct a detailed root cause analysis of any issues or anomalies observed by thinking step by step. 
        
        The metrics_pictures describes the pod-level data received bandwidth metric for each pod and attached at the bottom of the instruction, if you did not get any pictures, please tell me.

        The log_data has been pre-processed to capture a few important information: {log_text}    

        The metrics_summary data:
        {metrics_summary}

        The output is expected to contain these variables and structured in JSON format:
        1. The number of pictures you received
        2. A ranked list of tuple with size 20, with first element of tuple is the pod name, and the second element is the reason for it is likely to be the root cause. The first tuple in the list should be the most likely root cause pod. 
        3. Provide explaination of ranking.
        """

        self.logger.info("Benchmark Inferencing ...")

        return self.llm.llm_generate_text_input(
            self.config.llm_model,
            instruction,
            system,
            max_tokens=self.config.max_token,
        )

    def generate_time_analysis(self, relevant_data):
        system = "You are an experienced data scientist,You are very familiar with root cause analysis, which is a task of finding the root casue pod among all abnormal pods in an incident."
        instruction = f"""
        Based on the following data, determine whether the hypothesized time interval is reasonable.
        
        Rapid jumps in the data may indicate error times, while smooth and consistent data suggest normal times.

        Here are the data for the relevant pods:
        {relevant_data}

        Please analyze the data and determine if the hypothesized time interval is reasonable. If it is not reasonable, please indicate whether we should look an hour earlier or later, and explain your reasoning.
        
        The output is expected to contain these variables and structured in JSON format, respond with this JSON file only:
        1. root_cause_analysis_summerization_result
        2. top_hypothesis_pods: Top 10 most likely pods from the pods analyzed 
         2.1 pod_name: The pod name of this current top_hypothesis_pods.
         2.2 time_interval: The time interval of the hypothesis incident_time_interval. Example format: "2024-01-13T08:00:00.000Z/2024-01-13T09:00:00.000Z"
       
        
        """

        self.logger.info("Time analysis Inferencing ...")

        return self.llm.llm_generate_text_input(
            self.config.llm_model,
            instruction,
            system,
            max_tokens=self.config.max_token,
        )

    def generate_final_ranking(self, context):
        system = "You are an experienced data scientist. You are very familiar with root cause analysis, which is a task of finding the root casue pod among all abnormal pods in an incident."

        instruction = f"""Based on the initial root cause analysis summeriziation and root cause hypothesises, together with detailed analysis, provide the final root cause anayisis by thinking step by step. 
        
        The list of all pod names are in pod_names.
        
        The context of the previous results:
        {context}
        
        The output is expected to contain these variables in JSON format:
        1. A ranked list of tuple with size 20, with first element of tuple is the pod name, and the second element is the reason for it is likely to be the root cause. The first tuple in the list should be the most likely root cause node. 
        2. Provide explaination of ranking.
        """

        self.logger.info("Benchmark Inferencing ...")

        return self.llm.llm_generate_text_input(
            self.config.llm_model,
            instruction,
            system,
            max_tokens=self.config.max_token,
        )

