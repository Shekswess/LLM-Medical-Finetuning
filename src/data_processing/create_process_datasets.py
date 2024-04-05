import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from instruct_datasets import GemmaInstructDataset, MistralLlamaInstructDataset

REMOVE_COLUMNS = ["source", "focus_area"]
RENAME_COLUMNS = {"question": "input", "answer": "output"}
INSTRUCTION = "Answer the question truthfully."
DATASETS_PATHS = [
    r"D:\Work\LLM-7B-Medical-Finetuning\data\raw_data\medical_meadow_wikidoc.csv",
    r"D:\Work\LLM-7B-Medical-Finetuning\data\raw_data\medquad.csv",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_dataset_mistral_llama(dataset_path: str) -> pd.DataFrame:
    """
    Process the instruct dataset to be in the format required by the mistral or llama instruct model.
    :param dataset_path: The path to the dataset.
    :return: The processed dataset.
    """
    logger.info(
        f"Processing dataset: {dataset_path} for mistral or llama instruct model."
    )
    mistral_llama_dataset = MistralLlamaInstructDataset(dataset_path)
    mistral_llama_dataset.drop_columns(REMOVE_COLUMNS)
    logger.info("Columns removed!")
    mistral_llama_dataset.rename_columns(RENAME_COLUMNS)
    logger.info("Columns renamed!")
    mistral_llama_dataset.create_instruction(INSTRUCTION)
    logger.info("Instructions created!")
    mistral_llama_dataset.drop_bad_rows(["input", "output"])
    logger.info("Bad rows dropped!")
    mistral_llama_dataset.create_prompt()
    logger.info("Prompt column created!")
    return mistral_llama_dataset.get_dataset()


def process_dataset_gemma(dataset_path: str) -> pd.DataFrame:
    """
    Process the instruct dataset to be in the format required by the gemma instruct model.
    :param dataset_path: The path to the dataset.
    :return: The processed dataset.
    """
    logger.info(f"Processing dataset: {dataset_path} for gemma instruct model.")
    gemma_dataset = GemmaInstructDataset(dataset_path)
    gemma_dataset.drop_columns(REMOVE_COLUMNS)
    logger.info("Columns removed!")
    gemma_dataset.rename_columns(RENAME_COLUMNS)
    logger.info("Columns renamed!")
    gemma_dataset.create_instruction(INSTRUCTION)
    logger.info("Instructions created!")
    gemma_dataset.drop_bad_rows(["input", "output"])
    logger.info("Bad rows dropped!")
    gemma_dataset.create_prompt()
    logger.info("Prompt column created!")
    return gemma_dataset.get_dataset()


def create_dataset_hf(dataset: pd.DataFrame) -> DatasetDict:
    """
    Create a Hugging Face dataset from the pandas dataframe.
    :param dataset: The pandas dataframe.
    :return: The Hugging Face dataset.
    """
    return DatasetDict({"train": Dataset.from_pandas(dataset)})


if __name__ == "__main__":
    processed_data_path = r"D:\Work\LLM-7B-Medical-Finetuning\data\processed_data"
    os.makedirs(processed_data_path, exist_ok=True)

    mistral_llama_datasets = []
    gemma_datasets = []
    for dataset_path in DATASETS_PATHS:
        dataset_name = dataset_path.split(os.sep)[-1].split(".")[0]
        mistral_llama_dataset = process_dataset_mistral_llama(dataset_path)
        gemma_dataset = process_dataset_gemma(dataset_path)
        mistral_llama_datasets.append(mistral_llama_dataset)
        gemma_datasets.append(gemma_dataset)
        mistral_llama_dataset = create_dataset_hf(mistral_llama_dataset)
        gemma_dataset = create_dataset_hf(gemma_dataset)
        mistral_llama_dataset.push_to_hub(
            f"mistral_llama_{dataset_name}_instruct_dataset"
        )
        gemma_dataset.push_to_hub(f"gemma_{dataset_name}_instruct_dataset")

    mistral_llama_dataset = pd.concat(mistral_llama_datasets, ignore_index=True)
    gemma_dataset = pd.concat(gemma_datasets, ignore_index=True)

    mistral_llama_dataset = create_dataset_hf(mistral_llama_dataset)
    gemma_dataset = create_dataset_hf(gemma_dataset)

    mistral_llama_dataset.save_to_disk(
        os.path.join(processed_data_path, "medical_mistral_llama_instruct_dataset")
    )
    gemma_dataset.save_to_disk(
        os.path.join(processed_data_path, "medical_gemma_instruct_dataset")
    )

    mistral_llama_dataset.push_to_hub("medical_mistral_llama_instruct_dataset")
    gemma_dataset.push_to_hub("medical_gemma_instruct_dataset")
