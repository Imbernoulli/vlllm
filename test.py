import os
import json
import multiprocessing as mp
import shutil
from typing import List, Dict, Optional
from jload import jload, jsave
import torch
from vlite import generate, VLLMGeneratorError

MODEL_ID = "/home/test/test12/bohan/models/Llama-3.2-3B-Instruct"
DUMMY_JSONL_PATH = "test.jsonl"
MESSAGE_KEY = "problem_statement"
RESULT_KEY_PREFIX = "test_result_"

def run_test_scenario(
    test_name: str,
    input_data: List[Dict],
    model_id: str,
    message_key: str,
    result_key: str,
    system_prompt: Optional[str] = None,
    tp_size: int = 1,
    pp_size: int = 1,
    n_samples: int = 1,
    num_workers: int = 1,
    temperature_val: float = 0.1,
    use_sample_flag: bool = False,
    max_model_len_val: int = 2048,
    max_output_len_val: int = 1024,
    chunk_size_val: Optional[int] = None,
    gpu_assignments_val: Optional[List[List[int]]] = None,
    trust_remote_code_val: bool = True,
    gpu_memory_utilization_val: float = 0.80,
    dtype_val: str = "auto"
):
    """运行一个测试场景并打印结果"""
    print(f"\n>>> Starting test: {test_name}")
    print(f"Parameters: workers={num_workers}, n_samples={n_samples}, use_sample={use_sample_flag}, chunk_size={chunk_size_val}, system_prompt='{system_prompt is not None}'")

    data_copy = [item.copy() for item in input_data]

    processed_data = generate(
        model_id=model_id,
        data=data_copy,
        system=system_prompt,
        message_key=message_key,
        tp=tp_size,
        pp=pp_size,
        n=n_samples,
        worker_num=num_workers,
        temperature=temperature_val,
        use_sample=use_sample_flag,
        result_key=result_key,
        max_model_len=max_model_len_val,
        max_output_len=max_output_len_val,
        chunk_size=chunk_size_val,
        gpu_assignments=gpu_assignments_val,
        trust_remote_code=trust_remote_code_val,
        gpu_memory_utilization=gpu_memory_utilization_val,
        dtype=dtype_val
    )
    jsave(processed_data, f"{result_key}.json")


if __name__ == "__main__":


    initial_data = jload(DUMMY_JSONL_PATH)

    num_test_workers = 2

    run_test_scenario(
        test_name=f"Multi-Worker ({num_test_workers} workers)",
        n_samples=8,
        input_data=initial_data,
        model_id=MODEL_ID,
        message_key=MESSAGE_KEY,
        result_key=RESULT_KEY_PREFIX + "multi_worker",
        num_workers=num_test_workers,
        chunk_size_val = 25
    )

    print("\nAll tests completed.")
