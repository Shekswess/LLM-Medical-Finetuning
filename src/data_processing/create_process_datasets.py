import pandas as pd
import os
from datasets import Dataset, DatasetDict
from instruct_datasets import GemmaInstructDataset, MistralLlamaInstructDataset

REMOVE_COLUMNS = ['source', 'focus_area']
RENAME_COLUMNS = {'question': 'input', 'answer': 'output'}
INSTRUCTION = "Answer the question truthfully."
DATASETS_PATHS = [r"D:\Work\LLM-7B-Medical-Finetuning\data\medical_meadow_wikidoc.csv", r"D:\Work\LLM-7B-Medical-Finetuning\data\medquad.csv"]


def process_dataset_mistral_llama(dataset_path: str) -> pd.DataFrame:
    mistral_llama_dataset = MistralLlamaInstructDataset(dataset_path)
    mistral_llama_dataset.drop_columns(REMOVE_COLUMNS)
    mistral_llama_dataset.rename_columns(RENAME_COLUMNS)
    mistral_llama_dataset.create_instruction(INSTRUCTION)
    return mistral_llama_dataset.get_dataset() 


def process_dataset_gemma(dataset_path: str) -> pd.DataFrame:
    gemma_dataset = GemmaInstructDataset(dataset_path)
    gemma_dataset.drop_columns(REMOVE_COLUMNS)
    gemma_dataset.rename_columns(RENAME_COLUMNS)
    gemma_dataset.create_instruction(INSTRUCTION)
    return gemma_dataset.get_dataset()


if __name__ == "__main__":
    processed_data_path = r"D:\Work\LLM-7B-Medical-Finetuning\src\processed_data"
    os.makedirs(processed_data_path, exist_ok=True)
    mistral_llama_datasets = [process_dataset_mistral_llama(dataset_path) for dataset_path in DATASETS_PATHS]
    gemma_datasets = [process_dataset_gemma(dataset_path) for dataset_path in DATASETS_PATHS]
    mistral_llama_dataset = pd.concat(mistral_llama_datasets, ignore_index=True)
    gemma_dataset = pd.concat(gemma_datasets, ignore_index=True)
    mistral_llama_dataset = DatasetDict({'train': Dataset.from_pandas(mistral_llama_dataset)})
    print(mistral_llama_dataset)
    # gemma_dataset = Dataset.from_pandas(gemma_dataset)
    mistral_llama_dataset.save_to_disk(os.path.join(processed_data_path, "medical_mistral_llama_instruct_dataset"))
    # gemma_dataset.save_to_disk(os.path.join(processed_data_path, "medical_gemma_instruct_dataset"))
    # mistral_llama_dataset.push_to_hub("mistral_llama_dataset")