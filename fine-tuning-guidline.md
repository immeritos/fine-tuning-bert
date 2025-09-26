# Supervised Fine-Tuning

> This document covered essential aspects of LLM fine-tuning, both in theory and practice, including: Creating instruction dataset, SFT techniques and pratical implementation.

## Creating a high-quality instruction dataset
### Creating an instruction dataset
Instruction datasets are defined as pairs of instructions and answers. The instructions are the inputs of the model, used as context during fine-tuning. The answers are the expected outputs of the model.

#### Data quantity
High-quality data can be described through three main dimensions:
1. Accuracy
2. Diversity
3. Complexity

We can distinguish two types of finetunes: **general-purpose**, aimed to reproduce the capabilities of models like GPT, and **task- or domain-specific** models, designed to optimize their performance for a particular application.
-  General-purpose models cover more topics, which requires additional samples. Based on the quality of these finetunes, we recommend an instruction dataset of at least **one million** samples to create a good general-purpose instruct model.
- Task-specific models are designed to excel at a particular function, such as translation, summarization, or sentiment analysis. The data required for task-specific fine-tuning is generally more manageable, ranging from **100 to 100,000** samples. 
-  Domain-specific models, on the other hand, aim to tweak the LLM with specialized knowledge and familiarity with the vocabulary and linguistic patterns of a particular field. The data requirements for domain-specific fine-tuning can vary widely depending on the complexity and breadth of the domain.

Rule-based filtering is a systematic approach to data quality control that relies on explicit, predefined rules to evaluate and filter data samples. 
1. Length filtering
2. Keyword exclusion
3. Format checking

#### Data deduplication
To deduplicate datasets, we distinguish between exact and fuzzy deduplication. 

Exact deduplication removes identical samples through a straightforward process involving data normalization, hash generation, and duplicate removal. 

The most popular approach to fuzzy deduplication is **MinHash deduplication**. MinHash operates by generating compact representations, or signatures, for each data item.  In practice, MinHash transforms data items (such as text documents) into sets of shingles, applies multiple hash functions to these sets, and selects the minimum hash values to form signature vectors.

**Semantic similarity** takes a different approach by 
focusing on the meaning of text for deduplication. This method involves converting words or entire samples into vector representations using various natural language processing techniques. Word embedding models such as Word2Vec, GloVe, and FastText transform individual words into dense vectors, capturing semantic relationships.

Once these vector representations are obtained, deduplication can be performed by comparing the **similarity between vectors**. Common similarity measures include cosine similarity or Euclidean distance. 

Methods like **K-means**, **DBSCAN**, or **hierarchical clustering** can efficiently organize the vector space, allowing for the identification of clusters that represent semantically similar content. Within each cluster, a representative sample can be retained while others are marked as duplicates.

#### Data decontamination
Data decontamination is the process of ensuring that the training dataset does not contain samples that are identical or highly similar to those in the evaluation or test sets. This step is important for ensuring the quality of the model evaluation and preventing overfitting or memorization of test data.

**Exact matching** can be used to remove any training samples that are identical to those in the evaluation sets. This can be done using hash functions or direct string comparisons. 

Next, we can also use **near-duplicate detection** methods to identify and remove training samples that are very similar to evaluation samples, even if they are not exactly the same. This often involves techniques like MinHash or computing similarity scores based on n-grams or embeddings.

#### Data quality evaluation
- **LLM-as-a-judge**: using multiple LLMs as a jury reduce bias and improve consistency
- **Reward models**: This model outputs multiple scores to target specific dimensions, such as helpfulness, correctness, coherence, complexity, and verbosity.
- **Classifiers or encoder-only models**

#### Data exploration
Data exploration is about understanding the dataset’s characteristics, strengths, and potential shortcomings.
- **Manual dataset exploration**: stratified sampling (selecting diverse samples), systematic review (using a criteria checklist), and collaborative review (involving multiple reviewers)
- **Statistical analysis**: NLTK or spaCy for tokenization and analysis of large text volumes; Matplotlib or Seaborn create histograms and word clouds
- **Topic clustering**: UMAP for dimensionality reduction, and 
DBSCAN for clustering

#### Data generation
When the available instruction datasets are not sufficient, creating custom data becomes necessary. **Synthetic data generation** using LLMs offers a more efficient and scalable alternative.

Well-crafted prompts can guide the language model to produce diverse, relevant, and high-quality instruction-response pairs.

An important aspect of synthetic data generation is the ability to control various attributes of the generated data. Furthermore, synthetic data generation can be particularly useful for addressing biases and gaps in existing datasets.

####  Data augmentation
- In-depth evolving focuses on enhancing the complexity of existing instructions.
- In-breadth evolving, on the other hand, aims to expand the diversity of the instruction dataset. 

#### Creating our own instruction dataset
Using the LLMOps pipeline:
1. Install openai, datasets, and tqdm
2. Create a Hugging Face dataset from raw data (JSON file)
3. Data cleaning
4. Chunk
5. Use an LLM to transform the articles into pairs of instructions and answers.
 - The user prompt
```python
def generate_instruction_answer_pairs(
extract: str, client: OpenAI
) -> List[Tuple[str, str]]:
prompt = f"""Based on the following extract, generate five 
instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. 
each answer \
must provide a relevant paragraph based on the information found in 
the \
context. Only use concepts from the context to generate the 
instructions. \
Instructions must never explicitly mention a context, a system, a 
course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that 
mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements 
into a language model. \
The idea is to create a digital replica of your writing habits using 
advanced AI techniques. \
Provide your response in JSON format with the following structure:
{{
"instruction_answer_pairs": [
{{"instruction": "...", "answer": "..."}},
...
]
}}
Extract:
{extract}
"""
```
 - the system prompt
```python
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", "content": "You are a helpful 
assistant who \
            generates instruction-answer pairs based on the given 
context. \
            Provide your response in JSON format.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1200,
        temperature=0.7,
    )
    # Parse the structured output
    result = InstructionAnswerSet.from_json(completion.choices[0].
 message.content)
    # Convert to list of tuples
    return result.pairs
```
6. Create a main function to automate the process
7. Create a main function to orchestrate the entire pipeline

See https://huggingface.co/datasets/mlabonne/llmtwin for the example.

## SFT techniques
SFT consists of re-training pre-trained models on a smaller dataset composed of pairs of instructions and answers. The goal of SFT is to turn a base model, which can only perform next-token prediction, into a useful assistant, capable of answering questions and following instructions. 

### When to fine-tune
SFT leverages pre-existing knowledge in the base model’s weights and refocuses the parameters for a specific purpose. 

In most scenarios, it is recommended to start with **prompt engineering** instead of directly fine-tuning models. Prompt engineering can be used with either open-weight or closed-source models. By using techniques like **few-shot prompting** or **retrieval augmented generation (RAG)**, numerous problems can efficiently be tackled without SFT.

**Advantages**:
1. SFT answers common needs in terms of control (“know your data”) and customizability (the fine-tuned model is unique). 
2. Fine-tuning allows developers to create more diverse interactions with LLMs, like tool analytics, moderation, and additional context.

**Disadvantages**:
1. Knowledge that is too distant from what has been learned in the pre-training set (such as an unknown or rare language) can be difficult to learn effectively. 
2. Fine-tuning a model on new knowledge could result in more frequent hallucinations.
3. "Catastrophic forgetting"

#### Chat templates
Once the instruction-answer pairs are parsed from the dataset format, we want to structure them in a chat template. Chat templates offer a unified way to present the instructions and answers to the model.

#### Parameter-efficient fine-tuning techniques
##### Full fine-tuning
Full fine-tuning refers to the most straightforward SFT technique, consisting of re-training every parameter in the base model. 

This method often provides the best results but requires significant computational resources. Using a single-GPU setting, the memory required can be estimated by the following factors:
- Parameters
- Gradients
- Optimizer States
- Activations

Several techniques can be employed to reduce memory usage during LLM fine-tuning:
- Model parallelism
- Gradient accumulation: 1 -> 8 ->16
- Memory-efficient optimizers
- Activation checkpointing

##### LoRA
LoRA is a parameter-efficient technique for fine-tuning LLMs. This is achieved by introducing trainable **low-rank matrices** that modify the behavior of the model without changing its original parameters.

At its core, LoRA employs a low-rank decomposition technique to update model weights efficiently. LoRA adds the two trainable matrices `A` and `B` and keeps the pre-trained weights `w` frozen.

The dimensions of matrices `A` and `B` are chosen such that their product has the same shape as `W` , but with a much lower rank.

LoRA comes with two hyperparameters:
1. **Rank($r$)**: Determines the size of the LoRA matrices. A common starting point is r=8 , but values up to 256 have shown good results in some cases. Larger ranks may capture more diverse tasks but could lead to overfitting.
2. **Alpha($\alpha$)**: A scaling factor applied to the LoRA update. In practice, we update the frozen weights $W$  by a factor of $\alpha$. This is why a common heuristic is to set $\alpha$ to twice the value of $r$, effectively applying a scaling factor of 2 to the LoRA update. You can experiment with different ratios in case of overfitting or underfitting.

It is possible to add a drop-out layer to prevent overfitting. The dropout rate is usually set between 0 and 0.1 as an optional regularization factor.

Initially, LoRA was primarily focused on **modifying the attention mechanism**, specifically the query (Q) and value (V) matrices in transformer layers. However, experiments have demonstrated significant benefits in extending LoRA’s application to other key components of the model:
- Key (K) matrices in attention layers
- Output projection layers (often denoted as O) in attention mechanisms
- Feed-forward or Multi-Layer Perceptron (MLP) blocks between attention layers
- Linear output layers

Multiple sets of LoRA weights can be combined for different tasks or domains, allowing flexible deployment and task switching without retraining. 

##### QLoRA
By combining quantization techniques with LoRA, QLoRA allows developers to fine-tune models on relatively small, widely available GPUs.

Like LoRA, instead of updating all model parameters during fine-tuning, QLoRA introduces small, trainable low-rank matrices (adapters) to specific layers of the model. To further reduce memory usage, QLoRA employs **double quantization**, which quantizes the quantization constants themselves. Additionally, it uses **paged optimizers** to manage memory spikes during training by leveraging Nvidia’s unified memory feature.

However, this memory efficiency comes at the cost of increased training time, with QLoRA being about 30% slower than LoRA. In terms of model performance, QLoRA shows only minor differences compared to LoRA.

#### Training parameters
- **Learning rate and scheduler**: 
Learning rate controls how much the model’s parameters are updated during training. 
The learning rate scheduler typically starts with a higher learning rate to enable rapid initial progress, then gradually decreases it in later stages to fine-tune the model more precisely. 
- **Batch size** : 
The batch size determines the number of samples processed before the model’s weights are updated. Typical batch sizes for LLM fine-tuning range from 1 to 32, with common values being 1, 2, 
4, 8, or 16.
Larger batch sizes generally lead to more stable gradient estimates and can improve training speed, as they provide a better approximation of the true gradient of the entire dataset. However, they also require more memory, which can be a limiting factor on GPUs with less VRAM. 
- **Maximum length and packing**
The maximum sequence length determines the longest input the model can process. It’s typically set between 512 and 4,096 tokens but can go up to 128,000 or more, depending on the task and available GPU memory. 
Truncation can occur at the beginning (left truncation) or end (right truncation) of the sequence. 

Packing maximizes the utilization of each training batch. Instead of assigning one sample per batch, packing combines multiple smaller samples into a single batch, effectively increasing the amount of data processed in each iteration.

- **Number of epochs**:
For LLM fine-tuning, the typical range is 1 to 10 epochs, with many successful runs using 2 to 5 epochs. The optimal number depends on factors such as task complexity, dataset size, and model architecture. More epochs allow the model to refine its learning, potentially improving performance. 

- **Optimizers**:
Optimizers adjust the model’s parameters to minimize the loss function. For LLM fine-tuning, AdamW (Adaptive Moment Estimation with Weight Decay) is highly recommended, particularly its 8-bit version.

- **Weight decay**:
Weight decay works by adding a penalty for large weights to the loss function, encouraging the model to learn simpler, more generalizable features. This helps the model avoid relying too heavily on any single input feature, which can improve its performance on unseen data. Typically, weight decay values range from 0.01 to 0.1, with 0.01 being a common starting point. For example, if you’re using the AdamW optimizer, you might set the weight decay to 0.01

- **Gradient checkpointing**:
Gradient checkpointing is a technique that reduces memory consumption during training by storing only a subset of intermediate activations generated in the forward pass.
## Implementing fine-tuning in practice
There are specialized tools and libraries to fine-tune models. In particular, we recommend the 
following:
- `TRL`: a library created and maintained by Hugging Face to train LLMs using SFT and preference alignment.
- `Axolotl`: streamlines the fine-tuning of LLMs with reusable 
YAML configuration files
- `Unsloth`: uses custom kernels to speed up training (2-5x) and reduce memory use (up to 80% less memory)

1. Load the model to fine-tune and its corresponding tokenizer
2. Define our LoRA configuration
3. Prepare the data in the right format for fine-tuning
4. Format this data using a chat template
5. Divide the dataset into training (95%) and test (5%) sets
6. Train the model
7. Test the fine-tuned model with a quick example
8. Save the model

Three of these metrics are important to monitor:
- **Training loss**: The loss should continuously decrease on average, indicating improving performance.
- **Validation loss**: A well-fitted model typically shows both training and validation losses decreasing and eventually stabilizing, with a small gap between them.
- **Gradient norm**: A stable or decreasing gradient norm generally means that the model is converging toward a local 
optimum.