import os


def create_dataset(base_path='./repos'):
    """
    Traverses through given base_path to collect information about each file in each repository,
    excluding files in known build output directories (bin, obj, dist, node_modules, etc.).
    
    Parameters:
    - base_path (str): Path to the directory containing all repositories.
    
    Returns:
    - List[dict]: A list of dictionaries, each containing file details.
    """
    dataset = []
    excluded_dirs = {'bin', 'obj', 'dist', 'node_modules'}  # Add more directories to exclude as needed

    for root, dirs, files in os.walk(base_path, topdown=True):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        if root.count(os.sep) - base_path.count(os.sep) >= 1:
            repo_name = root.split(os.sep)[
                root.split(os.sep).index('repos') + 1]  # Updated to correctly identify the repo name

            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, start=base_path)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file_content:
                        content = file_content.read()
                        dataset.append({
                            'file_name': file,
                            'repository_name': repo_name,
                            'relative_path': relative_path,
                            'content': content
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return dataset


# Use this function to gather your dataset
dataset = create_dataset()

print(dataset)

# # Assuming 'repo_data' is your dataset from Part 1
# preprocessed_data = [
#     {
#         'input_text': f"File: {d['file_name']}\nPath: {d['relative_path']}\nRepo: {d['repository_name']}",
#         'target_text': d['content']
#     } for d in repo_data
# ]

# # Here, you would convert 'preprocessed_data' to a format appropriate for your model, 
# # potentially a .json file or a direct input to a fine-tuning library.

# !pip install transformers torch

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "CodeLlama-7B-Instruct-GGUF"  # This is hypothetical; replace with the actual model name or path
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Example fine-tuning command - tailor this to the specific needs and API of your model and framework
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
# )

# trainer.train()
