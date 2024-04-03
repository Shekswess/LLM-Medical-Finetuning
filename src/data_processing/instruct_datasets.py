from abc import ABC, abstractmethod

import pandas as pd


class InstructDataset(ABC):
    """
    Abstract class for creating Instruct Datasets
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the dataset
        :param dataset_path: The path to the dataset
        """
        self.dataset = None
        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str) -> None:
        """
        Load the dataset from the given path
        :param dataset_path: The path to the dataset
        :return: None
        """
        self.dataset = pd.read_csv(dataset_path)

    def rename_columns(self, columns: dict[str, str]) -> None:
        """
        Rename the columns of the dataset
        :param columns: A dictionary of the form {old_name: new_name}
        :return: None
        """
        self.dataset = self.dataset.rename(columns=columns)

    def drop_columns(self, columns: list[str]) -> None:
        """
        Drop the columns from the dataset
        :param columns: A list of column names to drop
        :return: None
        """
        drop_columns = [col for col in columns if col in self.dataset.columns]
        self.dataset = self.dataset.drop(columns=drop_columns)

    def drop_bad_rows(self, columns: list[str]) -> None:
        """
        Drop the rows which have bad values in the columns
        :param columns: A list of columns to check for bad values
        :return: None
        """
        self.dataset = self.dataset.dropna(subset=columns)
        self.dataset = self.dataset.drop_duplicates(subset=columns)

    def create_instruction(self, instruction: str) -> None:
        """
        Create an instruction column in the dataset
        :param instruction: The instruction to add to the dataset
        :return: None
        """
        self.dataset["instruction"] = instruction

    @abstractmethod
    def create_prompt(self) -> None:
        """
        Create the prompt column in the dataset
        :return: None
        """
        pass

    def get_dataset(self) -> pd.DataFrame:
        """
        Get the dataset
        :return: The dataset
        """
        return self.dataset


class MistralLlamaInstructDataset(InstructDataset):

    def create_prompt(self):
        """
        Create the prompt column in the dataset which will be used for
        """
        prompts = []
        for index, row in self.dataset.iterrows():
            prompt = f"""<s>[INST] {row['instruction']} This is the question: {row['input']} [/INST] \\n {row['output']}</s>"""
            prompts.append(prompt)
        self.dataset["prompt"] = prompts


class GemmaInstructDataset(InstructDataset):

    def create_prompt(self):
        """
        Create the prompt column in the dataset which will be used for
        """
        prompts = []
        for index, row in self.dataset.iterrows():
            prompt = f"<start_of_turn>user {row['instruction']} This is the question: {row['input']}<end_of_turn> \\n <start_of_turn>model {row['output']}"
            prompts.append(prompt)
        self.dataset["prompt"] = prompts
