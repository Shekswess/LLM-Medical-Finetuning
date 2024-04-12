# LLM-7B-Medical-Finetuning

This repository contains all the code necessary to fine-tune(PEFT using LoRA/QLoRa) the most popular 7B parameters instruct LLMs(Mistral, Llama, Gemma), specifically on medical data by utilizing. The code repository is based on two parts:
- preparing the instruct medical datasets
- fine-tuning the instruct LLMs on the prepared datasets

## Preparing the datasets

For this showcase project, two datasets are used:
- Medical meadow wikidoc (https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc/blob/main/README.md)
- Medquad (https://www.kaggle.com/datasets/jpmiller/layoutlm)

### Medical meadow wikidoc

The Medical Meadow Wikidoc dataset comprises question-answer pairs sourced from WikiDoc, an online platform where medical professionals collaboratively contribute and share contemporary medical knowledge. WikiDoc features two primary sections: the "Living Textbook" and "Patient Information". The "Living Textbook" encompasses chapters across various medical specialties, from which we extracted content. Utilizing GTP-3.5-Turbo, the paragraph headings are transformed into questions and utilized the respective paragraphs as answers. Notably, the structure of "Patient Information" is distinct; each section's subheading already serves as a question, eliminating the necessity for rephrasing.

### Medquad

MedQuAD is a comprehensive collection consisting of 47,457 medical question-answer pairs compiled from 12 authoritative sources within the National Institutes of Health (NIH), including domains like cancer.gov, niddk.nih.gov, GARD, and MedlinePlus Health Topics. These question-answer pairs span 37 distinct question types, covering a wide spectrum of medical subjects, including diseases, drugs, and medical procedures. The dataset features additional annotations provided in XML files, facilitating various Information Retrieval (IR) and Natural Language Processing (NLP) tasks. These annotations encompass crucial information such as question type, question focus, synonyms, Unique Identifier (CUI) from the Unified Medical Language System (UMLS), and Semantic Type. Moreover, the dataset includes categorization of question focuses into three main categories: Disease, Drug, or Other, with the exception of collections from MedlinePlus, which exclusively focus on diseases.

For our experiments there are 8 different versions of the datasets, available as Hugging Face datasets:
- gemma_medical_meadow_wikidoc_instruct_dataset -> Medical Meadow Wikidoc dataset for Gemma Instruct (https://huggingface.co/datasets/Shekswess/gemma_medical_meadow_wikidoc_instruct_dataset
- mistral_llama_medical_meadow_wikidoc_instruct_dataset -> Medical Meadow Wikidoc dataset for Mistral and Llama Instruct (https://huggingface.co/datasets/Shekswess/mistral_llama_medical_meadow_wikidoc_instruct_dataset
- gemma_medquad_instruct_dataset -> Medquad dataset for Gemma Instruct (https://huggingface.co/datasets/Shekswess/gemma_medquad_instruct_dataset)
- mistral_llama_medquad_instruct_dataset -> Medquad dataset for Mistral and Llama Instruct (https://huggingface.co/datasets/Shekswess/mistral_llama_medquad_instruct_dataset)
- medical_gemma_instruct_dataset -> combination of both Medical Meadow Wikidoc and Medquad dataset for Gemma Instruct (https://huggingface.co/datasets/Shekswess/medical_gemma_instruct_dataset)
- medical_mistral_llama_instruct_dataset -> combination of both Medical Meadow Wikidoc and Medquad dataset for Mistral and Llama Instruct (https://huggingface.co/datasets/Shekswess/medical_mistral_llama_instruct_dataset)
- medical_gemma_instruct_dataset_short -> combination of both Medical Meadow Wikidoc and Medquad dataset for Gemma Instruct, but with a smaller dataset size 3000 entries (https://huggingface.co/datasets/Shekswess/medical_gemma_instruct_dataset_short)
- medical_mistral_llama_instruct_dataset_short -> combination of both Medical Meadow Wikidoc and Medquad dataset for Mistral and Llama Instruct, but with a smaller dataset size 3000 entries (https://huggingface.co/datasets/Shekswess/medical_mistral_llama_instruct_dataset_short) 


## Fine-tuning the LLMs

The fine-tuning of the LLMs is based around PEFT(Parameter Efficient Fine-Tuning - Supervised Tuning) using LoRA/QLoRA. Because the resources on Google Colab are limited(T4 GPU), sparing resources is crucial. That's why 4 bit quantization models are used, which are available on Hugging Face by using the models available by unsloth(https://github.com/unslothai/unsloth). Also most of the code is based on the library provided by unsloth.
For the fine-tuning, the following models are used:
- gemma-7b-it-bnb-4bit
- unsloth/llama-2-7b-chat-bnb-4bit
- mistral-7b-instruct-v02-bnb-4bit

Much more details about the fine-tuning process can be found in the notebooks in the `src/finetuning_notebooks` folder.

Models trained using this codebase are available on Hugging Face:
- Gemma: Shekswess/gemma-7b-it-bnb-4bit-medical(https://huggingface.co/Shekswess/gemma-7b-it-bnb-4bit-medical)
- Llama: Shekswess/llama-2-7b-chat-bnb-4bit-medical(https://huggingface.co/Shekswess/llama-2-7b-chat-bnb-4bit-medical)
- Mistral: Shekswess/mistral-7b-instruct-v02-bnb-4bit-medical(https://huggingface.co/Shekswess/mistral-7b-instruct-v02-bnb-4bit-medical)

DISCLAIMER: The models are trained on a small dataset (only 3000 entries).

## Repository structure
```
.
├── .vscode                                                 # VSCode settings
│   └── settings.json                                       # Settings for the formatting of the code
├── data                                                    # Datasets used in the project
│   ├── processed_datasets                                  # Processed datasets
│   │   ├── medical_gemma_instruct_dataset                  # Processed dataset for the Gemma
│   │   ├── medical_gemma_instruct_dataset_short            # Processed dataset for the Gemma with a smaller dataset size
│   │   ├── medical_mistral_llama_instruct_dataset          # Processed dataset for the Mistral and Llama
│   │   └── medical_mistral_llama_instruct_dataset_short    # Processed dataset for the Mistral and Llama with a smaller dataset size
│   └── raw_data                                            # Raw datasets
│       ├── medical_meadow_wikidoc.csv                      # Medical Meadow Wikidoc dataset
│       └── medquad.csv                                     # Medquad dataset
├── src                                                     # Source code
│   ├── data_processing                                     # Data processing scripts
│   │   ├── create_process_datasets.py                      # Script to create processed datasets
│   │   ├── instruct_datasets.py                            # Defining the processing of the datasets to be in the instruct format
│   │   └── requirements.txt                                # Requirements for the data processing scripts
│   └── finetuning_notebooks                                # Notebooks for the fine-tuning of the LLMs
│       ├── gemma_7b_it_medical.ipynb                       # Notebook for the fine-tuning of the Gemma LLM
│       ├── llama_2_7b_chat_medical.ipynb                   # Notebook for the fine-tuning of the Llama LLM
│       └── mistral_7b_instruct_v02_medical.ipynb           # Notebook for the fine-tuning of the Mistral LLM
├── .gitignore                                              # Git ignore file
└── README.md                                               # README file (this file)
```