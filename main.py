import os
import argparse
import logging
import json
import re
import pandas as pd

from src.utils import AttrDict
from src.generate import Generation
from src.data_preprocessing_metric import metrics_data_preprocess
from src.data_preprocessing_log import all_file_parse, get_log_prompt
from src.data_preprocessing_test_case_log_deep_dive import process_all_files_in_folder
from src.DataAnalysisAgent import get_time_series_agent_message, analyze_plots_and_samples

import numpy as np
import base64


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="SherlockGPT")
    parser.add_argument("--config", default="config.json", help="Path to config file")

    args = parser.parse_args()
    # Set log level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Check if config file exists
    if not os.path.isfile(args.config):
        logger.error("Config file does not exist")
        return

    config = json.load(open(args.config))
    config = AttrDict(config)

    metric_summary_output = []
    metrics_agent = Generation(
        config,
        logger
    )
    logger.info("Loading metrics data...")

    if not os.path.exists(config.metrics_individual_analysis_file):
        frame_list = metrics_data_preprocess(config.metrics_path)
        logger.info("Metrics analysis generating...")
        for frame in frame_list:
            metric_summary_output.append(metrics_agent.generate_metrics_analysis(frame))
        with open("processed_data/metric_summary_output.txt", "w", encoding="utf-8") as f:
            for item in metric_summary_output:
                f.write(item + "\n")
        with open("processed_data/metric_summary_output.json", "w", encoding="utf-8") as f:
            json.dump(metric_summary_output, f, ensure_ascii=False, indent=4)
        metrics_summary = metric_summary_output

    else:
        logger.info("Metrics analysis loading...")
        with open(config.metrics_individual_analysis_file) as f:
            metrics_summary = json.load(f)
    #logger.info(metrics_summary)


    logger.info("Loading logs data...")
    if os.path.exists(config.log_summary_path):
        df_log = pd.read_csv(config.log_summary_path)
        _, logs_summary = get_log_prompt(df_log)
        #logger.info(logs_summary)
    else:
        logger.info("Preprocessing logs data...")
        df_log = all_file_parse(config.log_path)
        logs_summary = get_log_prompt(df_log)
    

    logger.info("Processing logs deepdive data...")
    if os.path.exists(config.log_plot_path):
        logger.info("Already processed deep dive data!")

    else:
        logger.info("Preprocessing logs data deep dive...")
        process_all_files_in_folder(config.log_path)
    



    if config.bench_mark_run is True:
        logger.info("Benchmark generation...")
        benchmark_agent = Generation(
            config,
            logger
            )
        benchmark_result = benchmark_agent.generate_benchmark(metrics_summary, 
                                                logs_summary)
        logger.info(f"{benchmark_result}")
        with open("processed_data/benchmark_result.txt", "w", encoding="utf-8") as f:
            for item in benchmark_result:
                f.write(item)

    else:
        logger.info("SherlockGPT run...")
        """
        # When we need to generate hypothesis
        logger.info("Hypothesis generation...")
        hypothesis_agent = Generation(
            config,
            logger
            )
        likely_hypothesis = hypothesis_agent.generate_likely_hypothesis(metrics_summary, 
                                                logs_summary)


        logger.info(f"{likely_hypothesis}")

        with open("processed_data/likely_hypothesis.txt", "w", encoding="utf-8") as f:
            for item in likely_hypothesis:
                f.write(item)


        logger.info("Hypothesis vaidation...")

        
        Input with hypo_1:{
                            incident_time_interval: time
                            root_cause_pod: pod_0
                            hypothesis_explanation: text
                            neighboring_pods: [pod_1, pod_2... pod_5]
                            },
                    hypo_2:....
        
        Do:
            1. 
        


        json_parts = likely_hypothesis.split('```json')
        if len(json_parts) > 1:
            json_part = json_parts[1].split('```')[0].strip()
            try:
                likely_hypothesis_json = json.loads(json_part)
                logger.info("Parsed JSON data:")
                logger.info(json.dumps(likely_hypothesis_json, indent=4))
            except json.JSONDecodeError as e:
                logger.info("Error parsing JSON:", e)

        hypotheses_time_analysis = {}

        time_analysis_agent = Generation(
            config,
            logger,
        )
        for idx, hypo in enumerate(likely_hypothesis_json['all_root_cause_hypothesis']):
            time_interval = hypo['incident_time_interval']
            root_cause_pod = hypo['root_cause_pod']
            neighboring_pods = hypo['neighboring_pods']

            change_rate_file = f"time-sequence-matrix/{time_interval}_change_rate.npy"
            sequence_file = f"time-sequence-matrix/{time_interval}_sequence.npy"

            change_rate_data = np.load(change_rate_file, allow_pickle=True)
            sequence_data = np.load(sequence_file, allow_pickle=True)

            change_rate_dict = {row[0]: row[1] for row in change_rate_data}
            sequence_dict = {row[0]: row[1] for row in sequence_data}
            
            pods = [root_cause_pod] + neighboring_pods

            relevant_data = {
                pod: {
                    'change_rate': change_rate_dict.get(pod),
                    'sequence': sequence_dict.get(pod)
                } for pod in pods
            }

            time_analysis_result = time_analysis_agent.generate_time_analysis(relevant_data)

            hypotheses_time_analysis[f'hypothesis_{idx}'] = {
                'time_analysis_result': time_analysis_result,
                'hypothesis': hypo
            }

            logger.info(f"LLM Analysis Result for hypothesis_{idx}:\n{time_analysis_result}\n")

        logger.info(f"{hypotheses_time_analysis}")
        with open("processed_data/hypotheses_time_analysis.txt", "w", encoding="utf-8") as f:
            json.dump(hypotheses_time_analysis, f, ensure_ascii=False, indent=4)
        """
        """
        Expected input
        1. Extract
            1.1 pod name
            1.2 time interval
    
        """

        # Get the message from the time-series analysis agent
        #time_series_message = get_time_series_agent_message()

        with open(config.likely_hypo_path, 'r') as file:
          likely_hypothesis = file.read()
        json_parts = likely_hypothesis.split('```json')
        if len(json_parts) > 1:
            json_part = json_parts[1].split('```')[0].strip()
            try:
                likely_hypothesis_json = json.loads(json_part)
                logger.info("Parsed JSON data:")
                logger.info(json.dumps(likely_hypothesis_json, indent=4))
            except json.JSONDecodeError as e:
                logger.info("Error parsing JSON:", e)

        # Analyze the plots based on the message
        analyze_plots_and_samples(likely_hypothesis_json, config.log_path, config.log_plot_path)

    logger.info("Mission completed!")


if __name__ == "__main__":
    main()
