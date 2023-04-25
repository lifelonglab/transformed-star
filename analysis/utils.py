from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


def process_eval_results(eval_results: pd.DataFrame) -> Dict:
    accuracy_dict = defaultdict(dict)
    for _, row in eval_results.iterrows():
        accuracy_dict[int(row['training_exp'])][int(row['eval_exp'])] = row['eval_accuracy']

    return accuracy_dict


def select_results(df: pd.DataFrame, scenario_type, model, multiple_results_resolution) -> Dict:
    results = defaultdict(lambda: defaultdict(dict))  # scenario: {strategy: metrics}
    for _, row in df.iterrows():
        if _process_df_should_add_results(row, results, scenario_type=scenario_type, model=model,
                                          multiple_results_resolution=multiple_results_resolution):
            metrics_dict = {
                'eval_results_accuracy': row['eval_results_accuracy'],
                'eval_results_bwt': row['eval_results_bwt'],
                'eval_results_fwt': row['eval_results_fwt'],
                'source': row['source']
            }
            results[row['scenario']][row['strategy']] = metrics_dict
    return results


def _process_df_should_add_results(row, results, scenario_type, model, multiple_results_resolution) -> bool:
    if row['scenario_type'] == scenario_type and row['model'] == model and row['eval_results_accuracy'] != -1:
        if len(results[row['scenario']][row['strategy']]) > 0:
            print("Resolving multiple results")
            if multiple_results_resolution == 'max_accuracy':
                current_accuracy = results[row['scenario']][row['strategy']]['eval_results_accuracy']
                return row['eval_results_accuracy'] > current_accuracy  # swap results
            else:
                print("Multiple results for given scenario type, model, scenario and strategy: error")
                exit(1)
        else:
            return True

    return False


icarl_augment_data = Compose([
    ToTensor(),
    # Resize((64, 64)),
    Normalize(mean=(0.9221,), std=(0.2681,))
])
