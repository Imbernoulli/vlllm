import os
import json
import torch
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Union

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class VLLMGeneratorError(Exception):
    """Custom exception for errors in the VLLM generator."""
    pass

def _prepare_llm_inputs(
    data_chunk: List[Dict],
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str],
    message_key: str,
    n_samples: int,
    use_vllm_sample_n: bool
) -> tuple[List[List[int]], List[int]]:
    """
    Prepares tokenized inputs for vLLM and maps them back to original items within the given data_chunk.

    Returns:
        all_prompt_token_ids: A flat list of token ID lists for vLLM.
        item_indices_map: A list where item_indices_map[i] is the index of the original item
                          in data_chunk that all_prompt_token_ids[i] corresponds to.
    """
    all_prompt_token_ids = []
    item_indices_map = []  # Maps each entry in all_prompt_token_ids to its original item's index in data_chunk

    for original_idx_in_chunk, item in enumerate(data_chunk): # original_idx_in_chunk is relative to this specific data_chunk
        message_content = item.get(message_key)
        if message_content is None:
            print(f"Warning: Message key '{message_key}' not found in item: {item}. Skipping.")
            continue

        constructed_messages = []
        if system_prompt:
            constructed_messages.append({"role": "system", "content": system_prompt})

        if isinstance(message_content, str):
            if message_content.strip(): 
                constructed_messages.append({"role": "user", "content": message_content})
            elif not system_prompt: 
                print(f"Warning: Empty message content and no system prompt for item: {item}. Skipping.")
                continue
        elif isinstance(message_content, list):
            valid_list_format = True
            has_user_message_in_list = False
            for msg in message_content:
                if not (isinstance(msg, dict) and "role" in msg and "content" in msg):
                    print(f"Warning: Invalid list format for message_key '{message_key}' in item: {item}. Expected list of {{'role': ..., 'content': ...}}. Skipping.")
                    valid_list_format = False
                    break
                if msg["role"] != "system" or not system_prompt:
                    constructed_messages.append(msg)
                if msg["role"] == "user":
                    has_user_message_in_list = True

            if not valid_list_format:
                continue
            if not has_user_message_in_list and not any(m.get("role") == "user" for m in constructed_messages):
                 print(f"Warning: No user messages found in list for item {item}, and no system prompt to form a valid request with user content. Skipping. Current messages: {constructed_messages}")
                 continue
        else:
            print(f"Warning: Invalid type for message_key '{message_key}' in item: {item}. Expected str or list. Skipping.")
            continue

        if not constructed_messages or all(msg.get("role") == "system" for msg in constructed_messages):
            print(f"Warning: No actionable messages (e.g., user prompt) to process for item: {item}. Skipping. Constructed: {constructed_messages}")
            continue
        
        if not any(msg.get("role") == "user" for msg in constructed_messages):
            if all(msg.get("role") == "system" for msg in constructed_messages):
                 print(f"Warning: Only system messages found for item {item}. Skipping as most models require a user prompt. Messages: {constructed_messages}")
                 continue

        current_messages_for_template = constructed_messages
        token_ids = tokenizer.apply_chat_template(
            current_messages_for_template,
            tokenize=True,
            add_generation_prompt=True
        )
        if not isinstance(token_ids, list) or not all(isinstance(tid, int) for tid in token_ids):
            if hasattr(token_ids, 'tolist'): 
                token_ids = token_ids.tolist()

        if not use_vllm_sample_n:
            for _ in range(n_samples):
                all_prompt_token_ids.append(token_ids)
                item_indices_map.append(original_idx_in_chunk) # Index within the current data_chunk
        else:
            all_prompt_token_ids.append(token_ids)
            item_indices_map.append(original_idx_in_chunk) # Index within the current data_chunk

    return all_prompt_token_ids, item_indices_map


def _process_outputs_and_update_items(
    vllm_outputs: List[Any],
    tokenizer: AutoTokenizer,
    original_items_chunk: List[Dict], # This is the pre-defined chunk that was processed
    item_indices_map: List[int], # Indices relative to original_items_chunk
    n_samples_per_prompt: int,
    use_vllm_sample_n: bool,
    result_key: str
):
    """
    Processes vLLM outputs and updates the original data items within the given chunk.
    original_items_chunk is modified in-place.
    """
    num_unique_items_in_chunk = len(original_items_chunk)
    temp_results_for_unique_items = [[] for _ in range(num_unique_items_in_chunk)]

    for i, request_output in enumerate(vllm_outputs):
        # item_indices_map[i] is the index within original_items_chunk
        original_item_idx_in_chunk = item_indices_map[i] 

        if original_item_idx_in_chunk >= num_unique_items_in_chunk :
            print(f"Warning: item_indices_map value {original_item_idx_in_chunk} is out of bounds for original_items_chunk (size {num_unique_items_in_chunk}). Skipping this output.")
            continue

        if use_vllm_sample_n:
            for sample_output in request_output.outputs:
                decoded_text = tokenizer.decode(sample_output.token_ids, skip_special_tokens=True)
                temp_results_for_unique_items[original_item_idx_in_chunk].append(decoded_text)
        else:
            if request_output.outputs:
                decoded_text = tokenizer.decode(request_output.outputs[0].token_ids, skip_special_tokens=True)
                temp_results_for_unique_items[original_item_idx_in_chunk].append(decoded_text)
            else:
                print(f"Warning: No output found for an input corresponding to original item index {original_item_idx_in_chunk} in its chunk.")
                temp_results_for_unique_items[original_item_idx_in_chunk].append("")

    for i in range(num_unique_items_in_chunk):
        current_item_results = temp_results_for_unique_items[i]
        if n_samples_per_prompt == 1:
            original_items_chunk[i][result_key] = current_item_results[0] if current_item_results else ""
        else:
            original_items_chunk[i][result_key] = current_item_results


def _generation_worker(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    model_id_str: str,
    system_prompt_str: Optional[str],
    message_key_str: str,
    tp_size_val: int, 
    pp_size_val: int,
    n_samples_val: int,
    temperature_val: float,
    use_vllm_sample_n_val: bool,
    result_key_str: str,
    max_model_len_val: int,
    max_output_len_val: int,
    # chunk_size_val is removed as worker no longer sub-chunks
    gpu_assignment_for_worker: Optional[List[int]], 
    trust_remote_code_val: bool,
    gpu_memory_utilization_val: float,
    dtype_val: str
):
    """Worker process for LLM generation. Processes one pre-defined chunk at a time."""
    
    if gpu_assignment_for_worker:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_assignment_for_worker))

    print(f"Worker {worker_id} (PID {os.getpid()}) started. "
          f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}', "
          f"LLM will be initialized with tensor_parallel_size: {tp_size_val}")
    
    processed_items_in_worker = 0

    tokenizer = AutoTokenizer.from_pretrained(model_id_str, trust_remote_code=trust_remote_code_val)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_args = {
        "model": model_id_str,
        "tensor_parallel_size": tp_size_val,
        "max_model_len": max_model_len_val,
        "trust_remote_code": trust_remote_code_val,
        "dtype": dtype_val,
    }

    if pp_size_val > 1:
        llm_args["pipeline_parallel_size"] = pp_size_val
        llm_args["distributed_executor_backend"] = "ray"
    else:
        llm_args["gpu_memory_utilization"] = gpu_memory_utilization_val
    
    vllm_engine = LLM(**llm_args)

    sampling_params_n_for_vllm = n_samples_val if use_vllm_sample_n_val else 1
    sampling_params = SamplingParams(
        n=sampling_params_n_for_vllm,
        temperature=temperature_val,
        max_tokens=max_output_len_val,
    )

    while True:
        try:
            task_data = task_queue.get(timeout=10) 
            if task_data is None: # Sentinel value
                break
            # data_partition is now the pre-defined chunk of items
            original_indices_start_of_chunk, data_partition_chunk = task_data 
        except mp.queues.Empty:
            continue
        except (EOFError, BrokenPipeError):
            print(f"Worker {worker_id} task queue closed, exiting.")
            break

        if not data_partition_chunk: # Should not happen if tasks are created properly
            result_queue.put((original_indices_start_of_chunk, [])) # Send back empty if received empty
            continue

        # Initialize result keys for items in this chunk before input prep
        for item_in_chunk in data_partition_chunk:
            if result_key_str not in item_in_chunk: # Ensure key exists
                item_in_chunk[result_key_str] = [] if n_samples_val > 1 else ""
        
        prompt_token_ids_list, item_indices_map = _prepare_llm_inputs(
            data_partition_chunk, tokenizer, system_prompt_str, message_key_str,
            n_samples_val, use_vllm_sample_n_val
        )

        if not prompt_token_ids_list: # All items in chunk might have been skipped
            # data_partition_chunk already has items with default/empty results initialized
            result_queue.put((original_indices_start_of_chunk, data_partition_chunk))
            continue
        
        vllm_raw_outputs = vllm_engine.generate(prompt_token_ids=prompt_token_ids_list, sampling_params=sampling_params)
        
        _process_outputs_and_update_items(
            vllm_raw_outputs, tokenizer, data_partition_chunk, item_indices_map,
            n_samples_val, use_vllm_sample_n_val, result_key_str
        )
        # data_partition_chunk items are updated in-place
        result_queue.put((original_indices_start_of_chunk, data_partition_chunk))
        processed_items_in_worker += len(data_partition_chunk)

    if vllm_engine is not None:
        del vllm_engine
    if 'tokenizer' in locals():
        del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Worker {worker_id} (PID {os.getpid()}) finished. Processed items: {processed_items_in_worker}.")


def generate(
    model_id: str,
    data: List[Dict],
    system: Optional[str] = None,
    message_key: str = "prompt",
    tp: int = 1, 
    pp: int = 1, 
    n: int = 1,  
    worker_num: int = 1,
    temperature: float = 0.7,
    use_sample: bool = False, 
    result_key: str = "results",
    max_model_len: int = 4096,
    max_output_len: int = 1024,
    chunk_size: Optional[int] = None,
    gpu_assignments: Optional[List[List[int]]] = None, 
    trust_remote_code: bool = True,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "auto"
) -> List[Dict]:
    """
    Generates text using vLLM. Data is pre-chunked if chunk_size is specified.
    Returns a new list with results. The input `data` list is not modified.
    """
    if not data:
        return []
    if worker_num < 1:
        raise VLLMGeneratorError("worker_num must be at least 1.")
    if n < 1:
        raise VLLMGeneratorError("n (number of samples) must be at least 1.")
    
    if pp > 1 and worker_num > 1:
        raise VLLMGeneratorError(
            "pipeline_parallel_size (pp) > 1 and worker_num > 1 cannot be used together. "
            "Set either pp=1 or worker_num=1."
        )

    if chunk_size is not None and chunk_size <= 0:
        print(f"Warning: chunk_size ({chunk_size}) is invalid, ignoring pre-chunking by chunk_size.")
        chunk_size = None # Treat as if not provided

    actual_gpu_assignments_per_worker: List[Optional[List[int]]] = [None] * worker_num
    effective_tp_per_worker: List[int] = [1] * worker_num

    if gpu_assignments:
        if len(gpu_assignments) != worker_num:
            raise VLLMGeneratorError(
                f"If gpu_assignments is provided, its length ({len(gpu_assignments)}) "
                f"must match worker_num ({worker_num})."
            )
        for i, assignment in enumerate(gpu_assignments):
            if not assignment or not all(isinstance(g, int) and g >= 0 for g in assignment):
                raise VLLMGeneratorError(
                    f"GPU assignment for worker {i} is invalid: {assignment}. "
                    "Must be a non-empty list of non-negative integers."
                )
            actual_gpu_assignments_per_worker[i] = sorted(list(set(assignment)))
            effective_tp_per_worker[i] = len(actual_gpu_assignments_per_worker[i])
            if effective_tp_per_worker[i] == 0 :
                 raise VLLMGeneratorError(f"GPU assignment for worker {i} resulted in zero GPUs.")
    else: 
        if tp < 1:
            raise VLLMGeneratorError("If gpu_assignments is not provided, tp (tensor_parallel_size per worker) must be at least 1.")
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            print(f"Warning: No GPUs available or detected by PyTorch. "
                  f"Each of {worker_num} workers will attempt to use tensor_parallel_size={tp}. ")
            for i in range(worker_num):
                effective_tp_per_worker[i] = tp
        else: 
            num_available_gpus = torch.cuda.device_count()
            required_gpus_total = worker_num * tp
            if num_available_gpus < required_gpus_total:
                raise VLLMGeneratorError(
                    f"Not enough GPUs available for {worker_num} workers, each requiring tp={tp} GPUs. "
                    f"Required total: {required_gpus_total}, Available: {num_available_gpus}."
                )
            current_gpu_idx = 0
            for i in range(worker_num):
                assigned_gpus_for_worker = list(range(current_gpu_idx, current_gpu_idx + tp))
                actual_gpu_assignments_per_worker[i] = assigned_gpus_for_worker
                effective_tp_per_worker[i] = tp
                current_gpu_idx += tp
    
    if mp.get_start_method(allow_none=True) is None:
        if os.name == 'posix':
            try:
                mp.set_start_method("spawn", force=False)
                print("Info: Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e:
                if mp.get_start_method() != "spawn":
                    print(f"Warning: Multiprocessing start method is '{mp.get_start_method()}', not 'spawn'. Error: {e}")
    
    task_q = mp.Queue()
    result_q = mp.Queue()
    
    num_data_items = len(data)
    indexed_data_copies = [(idx, item.copy() if isinstance(item, dict) else item) for idx, item in enumerate(data)]

    # --- Pre-chunking logic ---
    all_tasks_to_queue = []
    if chunk_size is not None: # chunk_size is validated to be > 0 or None
        print(f"Info: Pre-chunking data into chunks of size {chunk_size}.")
        current_offset_in_indexed_data = 0
        while current_offset_in_indexed_data < num_data_items:
            chunk_end_in_indexed_data = min(current_offset_in_indexed_data + chunk_size, num_data_items)
            
            # Get the (original_global_idx, item_copy) pairs for this specific chunk
            current_chunk_indexed_items = indexed_data_copies[current_offset_in_indexed_data : chunk_end_in_indexed_data]
            
            # The data for the task is the list of item_copies
            task_data_list = [item_copy for _, item_copy in current_chunk_indexed_items]
            
            # The original_start_idx for this task is the global index of its first item
            # This relies on indexed_data_copies[0][0] being 0, [1][0] being 1 etc.
            original_start_idx_of_chunk = indexed_data_copies[current_offset_in_indexed_data][0]
            
            all_tasks_to_queue.append((original_start_idx_of_chunk, task_data_list))
            current_offset_in_indexed_data += chunk_size
    else:
        # No global chunk_size: divide data into worker_num partitions (tasks)
        print(f"Info: Dividing data into {worker_num} partitions for workers (no pre-chunking by chunk_size).")
        partition_base_size = num_data_items // worker_num
        remainder = num_data_items % worker_num
        current_data_offset_for_partition = 0
        for i in range(worker_num):
            part_size = partition_base_size + (1 if i < remainder else 0)
            if part_size == 0:
                continue
            
            partition_end_offset = current_data_offset_for_partition + part_size
            current_partition_indexed_items = indexed_data_copies[current_data_offset_for_partition : partition_end_offset]
            task_data_list = [item_copy for _, item_copy in current_partition_indexed_items]
            
            if not current_partition_indexed_items: # Should not happen if part_size > 0
                 print(f"Warning: created an empty partition for worker {i}, skipping.")
                 continue

            original_start_idx_of_partition = current_partition_indexed_items[0][0]
            
            all_tasks_to_queue.append((original_start_idx_of_partition, task_data_list))
            current_data_offset_for_partition += part_size
    # --- End of Pre-chunking logic ---

    tasks_submitted_count = len(all_tasks_to_queue)
    if tasks_submitted_count == 0 and num_data_items > 0:
        print("Warning: No tasks were created for processing, though data was provided.")
        return [item_copy for _, item_copy in indexed_data_copies] # Return copies with no results key or default

    for task in all_tasks_to_queue:
        task_q.put(task)
    
    for _ in range(worker_num): # Sentinels for workers
        task_q.put(None)

    processes = []
    for i in range(worker_num):
        worker_gpu_assign = actual_gpu_assignments_per_worker[i]
        worker_tp_size = effective_tp_per_worker[i]

        p = mp.Process(
            target=_generation_worker,
            args=(
                i, task_q, result_q,
                model_id, system, message_key,
                worker_tp_size, pp, n, temperature, use_sample, result_key,
                max_model_len, max_output_len, # chunk_size_val removed from worker args
                worker_gpu_assign,
                trust_remote_code, gpu_memory_utilization, dtype
            )
        )
        p.start()
        processes.append(p)

    final_results_ordered = [None] * num_data_items 
    results_collected_count = 0
    
    active_tasks_expected = tasks_submitted_count 
    while results_collected_count < active_tasks_expected:
        try:
            result_package = result_q.get(timeout=300) 
            if result_package is None: 
                print("Warning: Received None from result queue unexpectedly.")
                continue

            original_start_idx_of_processed_chunk, processed_chunk_item_list = result_package
            
            for j, processed_item in enumerate(processed_chunk_item_list):
                global_idx_to_place = original_start_idx_of_processed_chunk + j
                if global_idx_to_place < num_data_items:
                    final_results_ordered[global_idx_to_place] = processed_item
                else:
                    print(f"Error: Result index {global_idx_to_place} out of bounds for final_results_ordered (size {num_data_items}).")
            results_collected_count += 1

        except mp.queues.Empty:
            print("Timeout waiting for results from workers. Checking worker status...")
            alive_workers = sum(1 for p_proc in processes if p_proc.is_alive())
            if alive_workers == 0 and results_collected_count < active_tasks_expected:
                print("!!! All workers seem to have exited prematurely. Aborting result collection.")
                for idx in range(num_data_items):
                    if final_results_ordered[idx] is None:
                        original_item_copy = indexed_data_copies[idx][1] if idx < len(indexed_data_copies) else {}
                        error_val = "ERROR_WORKER_FAILURE_NO_RESULT"
                        if not isinstance(original_item_copy, dict): original_item_copy = {"original_data": original_item_copy}
                        original_item_copy[result_key] = [error_val] * n if n > 1 else error_val
                        final_results_ordered[idx] = original_item_copy
                break 
            elif alive_workers > 0:
                print(f"{alive_workers} workers still alive. Continuing to wait...")
            else: 
                print("All workers exited and all expected results collected.")
                break 
        except Exception as e:
            print(f"Error collecting results: {type(e).__name__} - {e}")
            for idx in range(num_data_items):
                if final_results_ordered[idx] is None:
                    original_item_copy = indexed_data_copies[idx][1] if idx < len(indexed_data_copies) else {}
                    error_val = f"ERROR_RESULT_COLLECTION: {type(e).__name__}"
                    if not isinstance(original_item_copy, dict): original_item_copy = {"original_data": original_item_copy}
                    original_item_copy[result_key] = [error_val] * n if n > 1 else error_val
                    final_results_ordered[idx] = original_item_copy
            break

    for p_idx, p_proc in enumerate(processes):
        p_proc.join(timeout=60) 
        if p_proc.is_alive():
            print(f"Warning: Worker {p_idx} (PID {p_proc.pid}) did not terminate gracefully. Terminating.")
            p_proc.terminate()
            p_proc.join(timeout=10)
            if p_proc.is_alive():
                 print(f"Error: Worker {p_idx} (PID {p_proc.pid}) could not be terminated.")

    missing_data_count = sum(1 for item in final_results_ordered if item is None)
    if missing_data_count > 0:
        print(f"Warning: {missing_data_count} data items are missing (None) in the final result.")
        for idx in range(num_data_items):
            if final_results_ordered[idx] is None:
                original_item_copy = indexed_data_copies[idx][1] if idx < len(indexed_data_copies) else {}
                error_val = "ERROR_ITEM_UNPROCESSED_OR_LOST"
                if not isinstance(original_item_copy, dict): original_item_copy = { "original_data": original_item_copy }
                original_item_copy[result_key] = [error_val] * n if n > 1 else error_val
                final_results_ordered[idx] = original_item_copy
    
    return final_results_ordered