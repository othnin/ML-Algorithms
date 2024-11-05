# Levels of Using Large Language Models (LLMs)

## Level 1: Prompt Engineering

The first step in using LLMs effectively is prompt engineering—using an LLM “out of the box” without modifying model parameters. While some might overlook prompt engineering, it’s the most accessible way to work with LLMs, both technically and economically.

### Two Approaches to Prompt Engineering:

1. **The Easy Way: Using ChatGPT (or similar tools)**  
   - **Pros**: ChatGPT offers a simple, no-code way to interact with LLMs for free, making it incredibly user-friendly.
   - **Cons**: This approach lacks customization options (like setting temperature or response length) and doesn’t support automated, large-scale use cases.
   

2. **The Less Easy Way: Programmatic Interfaces**  
   - **Approach**: By interacting directly with an LLM via APIs (e.g., OpenAI’s API) or by running models locally (e.g., with the Transformers library), users gain control over parameters and scalability.
   - **Trade-offs**: This requires programming knowledge and may incur API costs, but it offers flexibility and customization. 
   [Here](Example Code prompt_engineering_chatgpt_v2.py)

While prompt engineering covers many use cases, some specific tasks may require a tailored model for optimal performance. In these cases, we move to the next level.

## Level 2: Model Fine-tuning

The next step is **fine-tuning**, which involves adjusting model parameters (weights and biases) for a particular use case. This is a form of transfer learning, where we build on an existing model’s knowledge.

### Steps for Fine-tuning:

1. **Obtain a Pre-trained LLM**: Start with a general-purpose LLM.
2. **Update Parameters with Labeled Data**: Fine-tune the model using a large set of labeled examples for the task.

Fine-tuning optimizes the model’s internal representations for specific tasks, enabling excellent performance with relatively modest data and computational needs. However, it requires more technical skill and computational resources than prompt engineering. 

#### Example Code: [Here](fine_tuning_peft2.ipynb)
In this example, we’ll use the Hugging Face ecosystem to fine-tune a language model for classifying text as either "positive" or "negative." We’ll work with distilbert-base-uncased, a model with around 70 million parameters based on BERT. Since this model was originally trained for language modeling, not classification, we apply transfer learning by replacing its original output layer with a classification head. We’ll also use LoRA (Low-Rank Adaptation) to optimize the fine-tuning process.

#### Example Code: [Here](fine_tuning_qlora.ipynb)
This repository contains a Python script that demonstrates how to fine-tune a large language model (LLM) using the Personalized Efficient Fine-Tuning (PEFT) approach. The script focuses on fine-tuning the Mistral-7B-Instruct-v0.2-GPTQ model, a 7B parameter instructional language model, for a text generation task.
The main functionality of the script includes:

Loading the pre-trained Mistral-7B-Instruct-v0.2-GPTQ model and tokenizer.
Preparing the model for k-bit training using the PEFT library.
Defining a LoRA (Low-Rank Adaptation) configuration to fine-tune the model.
Loading and tokenizing a dataset of YouTube comments.
Setting up the training pipeline using the Transformers library.
Training the fine-tuned model.
Generating a response based on a given input comment.

The script demonstrates how to leverage the PEFT library to efficiently fine-tune a large language model on a specific dataset, in this case, a collection of YouTube comments. The fine-tuned model can then be used for text generation tasks, such as providing responses to user queries.

## Level 3: Building Your Own LLM

For full customization, one can create an LLM from scratch. This involves defining all model parameters and training on a tailored dataset.

### Key Points:

- **Training Data**: High-quality, domain-specific data (e.g., medical texts for clinical models) is critical.
- **Flexibility vs. Resources**: Building a custom LLM provides unmatched flexibility but requires significant computational power, technical expertise, and a substantial budget—usually a large-scale project.
