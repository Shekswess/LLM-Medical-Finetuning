import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from instruct_datasets import GemmaInstructDataset, MistralLlamaInstructDataset

REMOVE_COLUMNS = ["source", "focus_area"]
RENAME_COLUMNS = {"question": "input", "answer": "output"}
INSTRUCTION = "Answer the question truthfully."
DATASETS_PATHS = [
    r"D:\Work\LLM-7B-Medical-Finetuning\data\medical_meadow_wikidoc.csv",
    r"D:\Work\LLM-7B-Medical-Finetuning\data\medquad.csv",
]
SEED = 42

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
    return gemma_dataset.get_dataset()


def split_dataset(dataset: pd.DataFrame, split_ratio: float) -> DatasetDict:
    """
    Split the dataset into train and test datasets.
    :param dataset: The dataset to split.
    :param split_ratio: The ratio of the dataset to be used for training.
    :return: The train and test datasets.
    """
    logger.info(
        f"Splitting dataset into train and test datasets with split ratio: {split_ratio}"
    )
    dataset = dataset.copy()
    dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_dataset = dataset[: int(split_ratio * len(dataset))]
    test_dataset = dataset[int(split_ratio * len(dataset)) :]
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_dataset),
            "test": Dataset.from_pandas(test_dataset),
        }
    )


if __name__ == "__main__":
    processed_data_path = r"D:\Work\LLM-7B-Medical-Finetuning\src\processed_data"
    os.makedirs(processed_data_path, exist_ok=True)

    mistral_llama_datasets = [
        process_dataset_mistral_llama(dataset_path)
        for dataset_path in DATASETS_PATHS
    ]
    gemma_datasets = [
        process_dataset_gemma(dataset_path) for dataset_path in DATASETS_PATHS
    ]

    mistral_llama_dataset = pd.concat(mistral_llama_datasets, ignore_index=True)
    gemma_dataset = pd.concat(gemma_datasets, ignore_index=True)

    mistral_llama_dataset = split_dataset(mistral_llama_dataset, 0.8)
    gemma_dataset = split_dataset(gemma_dataset, 0.8)

    mistral_llama_dataset.save_to_disk(
        os.path.join(processed_data_path, "medical_mistral_llama_instruct_dataset")
    )
    gemma_dataset.save_to_disk(
        os.path.join(processed_data_path, "medical_gemma_instruct_dataset")
    )

    mistral_llama_dataset.push_to_hub("medical_mistral_llama_instruct_dataset")
    gemma_dataset.push_to_hub("medical_gemma_instruct_dataset")
