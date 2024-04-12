import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from instruct_datasets import (
    GemmaInstructDataset,
    MistralInstructDataset,
    LlamaInstructDataset,
)

REMOVE_COLUMNS = ["source", "focus_area"]
RENAME_COLUMNS = {"question": "input", "answer": "output"}
INSTRUCTION = "Answer the question truthfully, you are a medical professional."
DATASETS_PATHS = [
    r"D:\Work\LLM-7B-Medical-Finetuning\data\raw_data\medical_meadow_wikidoc.csv",
    r"D:\Work\LLM-7B-Medical-Finetuning\data\raw_data\medquad.csv",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_dataset(dataset_path: str, model: str) -> pd.DataFrame:
    """
    Process the instruct dataset to be in the format required by the model.
    :param dataset_path: The path to the dataset.
    :param model: The model to process the dataset for.
    :return: The processed dataset.
    """
    logger.info(f"Processing dataset: {dataset_path} for {model} instruct model.")
    if model == "gemma":
        dataset = GemmaInstructDataset(dataset_path)
    elif model == "mistral":
        dataset = MistralInstructDataset(dataset_path)
    elif model == "llama":
        dataset = LlamaInstructDataset(dataset_path)
    else:
        raise ValueError(f"Model {model} not supported!")
    dataset.drop_columns(REMOVE_COLUMNS)
    logger.info("Columns removed!")
    dataset.rename_columns(RENAME_COLUMNS)
    logger.info("Columns renamed!")
    dataset.create_instruction(INSTRUCTION)
    logger.info("Instructions created!")
    dataset.drop_bad_rows(["input", "output"])
    logger.info("Bad rows dropped!")
    dataset.create_prompt()
    logger.info("Prompt column created!")
    return dataset.get_dataset()


def create_dataset_hf(
    dataset: pd.DataFrame, number_rows: int = None, shuffle: bool = False
) -> DatasetDict:
    """
    Create a Hugging Face dataset from the pandas dataframe.
    :param dataset: The pandas dataframe.
    :param number_rows: The number of rows to sample.
    :param shuffle: Whether to shuffle the dataset.
    :return: The Hugging Face dataset.
    """
    if shuffle:
        logger.info("Shuffling dataset!")
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    if number_rows:
        logger.info(f"Sampling {number_rows} rows!")
        dataset = dataset.iloc[:number_rows]
    dataset.reset_index(drop=True, inplace=True)
    return DatasetDict({"train": Dataset.from_pandas(dataset)})


if __name__ == "__main__":
    processed_data_path = r"D:\Work\LLM-7B-Medical-Finetuning\data\processed_data"
    os.makedirs(processed_data_path, exist_ok=True)

    mistral_datasets = []
    gemma_datasets = []
    llama_datasets = []
    for dataset_path in DATASETS_PATHS:
        dataset_name = dataset_path.split(os.sep)[-1].split(".")[0]

        mistral_dataset = process_dataset(dataset_path, "mistral")
        llama_dataset = process_dataset(dataset_path, "llama")
        gemma_dataset = process_dataset(dataset_path, "gemma")

        mistral_datasets.append(mistral_dataset)
        llama_datasets.append(llama_dataset)
        gemma_datasets.append(gemma_dataset)

        mistral_dataset = create_dataset_hf(mistral_dataset)
        llama_dataset = create_dataset_hf(llama_dataset)
        gemma_dataset = create_dataset_hf(gemma_dataset)

        mistral_dataset.push_to_hub(f"mistral_{dataset_name}_instruct_dataset")
        llama_dataset.push_to_hub(f"llama_{dataset_name}_instruct_dataset")
        gemma_dataset.push_to_hub(f"gemma_{dataset_name}_instruct_dataset")

    mistral_dataset = pd.concat(mistral_datasets, ignore_index=True)
    llama_dataset = pd.concat(llama_datasets, ignore_index=True)
    gemma_dataset = pd.concat(gemma_datasets, ignore_index=True)

    mistral_dataset = create_dataset_hf(mistral_dataset)
    llama_dataset = create_dataset_hf(llama_dataset)
    gemma_dataset = create_dataset_hf(gemma_dataset)

    mistral_dataset.save_to_disk(
        os.path.join(processed_data_path, "medical_mistral_instruct_dataset")
    )
    llama_dataset.save_to_disk(
        os.path.join(processed_data_path, "medical_llama_instruct_dataset")
    )
    gemma_dataset.save_to_disk(
        os.path.join(processed_data_path, "medical_gemma_instruct_dataset")
    )

    mistral_dataset.push_to_hub("medical_mistral_instruct_dataset")
    llama_dataset.push_to_hub("medical_llama_instruct_dataset")
    gemma_dataset.push_to_hub("medical_gemma_instruct_dataset")

    # Smaller datasets for free colab training
    mistral_dataset_short = pd.concat(mistral_datasets, ignore_index=True)
    llama_dataset_short = pd.concat(llama_datasets, ignore_index=True)
    gemma_dataset_short = pd.concat(gemma_datasets, ignore_index=True)

    mistral_dataset_short = create_dataset_hf(mistral_dataset_short, 3000, True)
    llama_dataset_short = create_dataset_hf(llama_dataset_short, 3000, True)
    gemma_dataset_short = create_dataset_hf(gemma_dataset_short, 3000, True)

    mistral_dataset_short.save_to_disk(
        os.path.join(
            processed_data_path, "medical_mistral_llama_instruct_dataset_short"
        )
    )
    llama_dataset_short.save_to_disk(
        os.path.join(processed_data_path, "medical_llama_instruct_dataset_short")
    )
    gemma_dataset_short.save_to_disk(
        os.path.join(processed_data_path, "medical_gemma_instruct_dataset_short")
    )

    mistral_dataset_short.push_to_hub("medical_mistral_instruct_dataset_short")
    llama_dataset_short.push_to_hub("medical_llama_instruct_dataset_short")
    gemma_dataset_short.push_to_hub("medical_gemma_instruct_dataset_short")
