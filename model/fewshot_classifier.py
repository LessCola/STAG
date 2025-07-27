import copy
import os
import re
from difflib import SequenceMatcher, get_close_matches

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT_TEMPLATE = {
    "classic": [
        (
            f"You are an AI assistant tasked with classifying input word sequences into one of the following categories: {'{classes}'}.\n"
            "You must choose strictly from these categories and no others.\n"
            "Each category has characteristic patterns shown in its examples.\n"
            "Here are examples of input sequences and their corresponding categories to guide you:\n\n"
        ),
        (
            "\nWhen given a new input sequence, identify its key patterns and match them to the most similar category from the examples.\n"
            "If no category is a clear match, choose the closest one.\n"
            "**IMPORTANT:** Output only the category name and nothing else."
        ),
    ],
    "simple": [
        (
            f"Classify input sequences into one of these categories: {'{classes}'}.\n"
            "Here are examples:\n\n"
        ),
        ("\nClassify the next input sequence. Output only the category name."),
    ],
}


class FewShotClassifier:
    def __init__(
        self,
        model_name,
        device,
        max_length=4096,
    ):
        self.model_id = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).half()
        if "llama-3" in self.model_id.lower():
            print(f"{self.model_id} used")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=True, padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

        else:
            print(f"{self.model_id} used")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=True, padding_side="left"
            )

            # Only add special tokens if they don't exist
            special_tokens = {}
            if self.tokenizer.bos_token != "<s>":
                special_tokens["bos_token"] = "<s>"
            if self.tokenizer.eos_token != "</s>":
                special_tokens["eos_token"] = "</s>"
            if self.tokenizer.pad_token is None:
                special_tokens["pad_token"] = "</s>"

            # Only update if we have new tokens to add
            if special_tokens:
                self.tokenizer.add_special_tokens(special_tokens)

            # Ensure config alignment
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

            # * re-apply padding side
            self.tokenizer.padding_side = "left"

        self.device = device  # "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()
        self.max_length = min(max_length, 4096)

    def set_classes(self, classes):
        # Store class names
        cleaned_classes = [
            re.sub(r"[^a-zA-Z0-9_\s]+", "", cls.lower().replace("_", " "))
            for cls in classes
        ]
        self.classes = cleaned_classes

        # Compute the maximum length in tokens of the class names
        class_name_lengths = [
            len(self.tokenizer.encode(class_name, add_special_tokens=False))
            for class_name in self.classes
        ]
        self.max_class_name_length = max(class_name_lengths)

        # Set output_length to the maximum class name length plus some buffer
        self.output_length = self.max_class_name_length + 5  # Adding a buffer

    def create_few_shot_prompt(self, few_shot_train_mask, tokens, categories):
        """
        Creates the conversation prompt with the system prompt and examples,
        # ! ensuring it does not exceed the maximum length and that examples are balanced and shuffled.

        Args:
            # graph: The graph containing the data.
            few_shot_train_mask (torch.Tensor): The few-shot training mask.
            tokens (list of list of str): The token sequences.
            categories (list of str): The corresponding categories.
        """
        # few_shot_train_mask = graph.ndata["few_shot_train_mask"].numpy()
        few_shot_train_mask = few_shot_train_mask.numpy()
        selected_classes = self.classes.copy()
        # number of shots
        self.shot = few_shot_train_mask.sum() // len(selected_classes)
        print(f"Number of shots: {self.shot}")
        np.random.shuffle(
            selected_classes
        )  # Shuffle classes to avoid initial order bias

        prompt_template = PROMPT_TEMPLATE["classic"]
        system_prompt = prompt_template[0]
        final_instruction = prompt_template[1]
        system_prompt = system_prompt.format(classes=", ".join(selected_classes))

        # print("system_prompt:", system_prompt)
        # print("final_instruction:", final_instruction)

        # Compute token lengths
        prompt_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=False)
        final_instruction_tokens = self.tokenizer.encode(
            final_instruction, add_special_tokens=False
        )
        buffer_tokens = 50  # Adjust buffer as needed

        # Compute maximum allowed tokens for examples
        max_prompt_length = self.max_length - buffer_tokens
        total_tokens = len(prompt_tokens) + len(final_instruction_tokens)

        # Collect examples and organize them by class
        selected_train_indices = np.where(few_shot_train_mask)[0]
        examples_by_class = {cls: [] for cls in selected_classes}
        for idx in selected_train_indices:
            category = categories[idx]
            category_cleaned = re.sub(
                r"[^a-zA-Z0-9_\s]+", "", category.lower().replace("_", " ")
            )
            if category_cleaned in examples_by_class:
                examples_by_class[category_cleaned].append(idx)

        # Shuffle examples within each class to avoid order bias
        for cls in examples_by_class:
            np.random.shuffle(examples_by_class[cls])

        # Initialize the examples list
        examples_list = []
        max_examples_per_class = max(
            len(indices) for indices in examples_by_class.values()
        )

        for i in range(max_examples_per_class):
            # Shuffle the classes for each iteration to avoid positional bias
            classes_shuffled = selected_classes.copy()
            np.random.shuffle(classes_shuffled)
            for cls in classes_shuffled:
                if i < len(examples_by_class[cls]):
                    examples_list.append(examples_by_class[cls][i])

        # Add examples until reaching the maximum prompt length
        examples_text = ""
        for idx in examples_list:
            input_seq = ", ".join(tokens[idx])
            category = categories[idx]
            # Clean category
            category_cleaned = re.sub(
                r"[^a-zA-Z0-9_\s]+", "", category.lower().replace("_", " ")
            )
            # ! the format of the example text will affect the performance of the model
            # example_text = f"- Input: {input_seq}\n Category: {category_cleaned}\n"
            example_text = (
                f"  Input: {input_seq}\n"
                f"  Category: {category_cleaned}\n"
                f"---\n"  # separator between examples
            )
            example_tokens = self.tokenizer.encode(
                example_text, add_special_tokens=False
            )
            example_tokens_length = len(example_tokens)

            # Check if adding this example would exceed max_prompt_length
            if total_tokens + example_tokens_length > max_prompt_length:
                print("Maximum prompt length reached.")
                break  # Stop adding examples if limit is reached
            else:
                examples_text += example_text
                total_tokens += example_tokens_length

        # Combine system prompt, examples, and final instruction
        prompt = system_prompt + examples_text + final_instruction
        print("Prompt:\n", prompt)

        # Begin the conversation with the system prompt
        conversation = [{"role": "system", "content": prompt}]

        # Store the conversation template
        self.conversation_template = conversation

    def build_conversation(self, input_sequence):
        """
        Builds the conversation for the given input sequence.

        Args:
            input_sequence (str): The input sequence to classify.

        Returns:
            prompt (str): The formatted conversation prompt.
        """
        # Copy the conversation template
        conversation = copy.deepcopy(self.conversation_template)

        # Add the new input sequence
        conversation.append(
            {"role": "user", "content": f"Input: {', '.join(input_sequence)}"}
        )

        # Build the prompt in the chat format expected by LLaMA-2 chat
        prompt = self.format_conversation(conversation)

        return prompt

    def format_conversation(self, conversation):
        """
        Formats the conversation in the way expected by LLaMA-2 chat.

        Args:
            conversation (list of dict): The conversation history.

        Returns:
            prompt (str): The formatted prompt.
        """
        if "vicuna" in self.model_id.lower():
            prompt = ""
            for turn in conversation:
                if turn["role"] == "system":
                    # Vicuna uses the system message as the first user message
                    prompt += f"USER: {turn['content']}\nASSISTANT: Understood. I will help classify the sequences.\n"
                elif turn["role"] == "user":
                    prompt += f"USER: {turn['content']}\nASSISTANT:"
                elif turn["role"] == "assistant":
                    prompt += f"{turn['content']}\n"
            return prompt
        elif "llama-2-7b" in self.model_id.lower():
            prompt = ""
            for turn in conversation:
                if turn["role"] == "system":
                    prompt += f"<s>[INST] <<SYS>>\n{turn['content']}\n<</SYS>>\n[/INST]"
                elif turn["role"] == "user":
                    prompt += f"<s>[INST] {turn['content']} [/INST]"
                elif turn["role"] == "assistant":
                    prompt += f"{turn['content']}</s>"
            return prompt
        else:  # "llama-2-13b"
            prompt = ""
            for turn in conversation:
                if turn["role"] == "system":
                    # Consistent system message formatting
                    prompt += f"<s>[INST] <<SYS>>\n{turn['content']}\n<</SYS>>\n\n[/INST]</s>\n"
                elif turn["role"] == "user":
                    # Consistent user message formatting
                    prompt += f"<s>[INST] {turn['content']} [/INST]</s>\n"
                elif turn["role"] == "assistant":
                    # Consistent assistant response formatting
                    prompt += f"<s>{turn['content']}</s>\n"
        return prompt

    # * deterministic
    def get_generation_config_deterministic(self):
        return {
            "max_new_tokens": self.output_length,
            "do_sample": False,  # Deterministic decoding
            "temperature": 0.0,  # No randomness
            "top_p": 1.0,  # Consider all tokens
            "early_stopping": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def classify(self, input_sequences, batch_size=8):
        results = []
        num_sequences = len(input_sequences)
        generation_config = self.get_generation_config_deterministic()

        for start_idx in range(0, num_sequences, batch_size):
            end_idx = min(start_idx + batch_size, num_sequences)
            batch_sequences = input_sequences[start_idx:end_idx]

            if "llama-3" in self.model_id.lower():
                # LLaMA-3 specific processing
                conversations = []
                for input_seq in batch_sequences:
                    conversation = copy.deepcopy(self.conversation_template)
                    conversation.append(
                        {
                            "role": "user",
                            "content": f"Input: {', '.join(input_seq)}",
                        }
                    )
                    conversations.append(conversation)

                input_ids = self.tokenizer.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                # manually created attention mask
                attention_mask = torch.ones_like(input_ids)

                # Update terminators for LLaMA-3
                generation_config["eos_token_id"] = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]

            else:
                # LLaMA-2 specific processing
                prompts = []
                for input_seq in batch_sequences:
                    prompt = self.build_conversation(input_seq)
                    prompts.append(prompt)

                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                ).to(self.device)

                input_ids = inputs["input_ids"]
                # attention_mask = inputs[
                #     "attention_mask"
                # ]  # Use the tokenizer's attention mask
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
                # attention_mask = torch.ones_like(input_ids) # use this as for LLaMA-3

            # Generate responses using common config
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config,
                )
            # Process outputs
            for i in range(len(batch_sequences)):
                input_length = attention_mask[i].sum().item()
                output = outputs[i]
                response = output[-(outputs.shape[-1] - input_length) :]
                generated_text = self.tokenizer.decode(
                    response, skip_special_tokens=True
                ).strip()
                predicted_category = self.parse_model_output(generated_text)
                # print("Generated text:", generated_text)
                # print("Predicted category:", predicted_category)
                # print("\n")
                results.append(predicted_category)

        return results

    def map_generated_category_to_class(self, generated_category):
        """
        Maps the generated category name to the closest known class name using multiple matching strategies.

        Args:
            generated_category (str): The category name generated by the model.
            classes (List[str]): The list of known class names.

        Returns:
            matched_class (str or None): The matched class name, or None if no match is found.
        """
        # Clean the generated category name
        cleaned_category = generated_category.strip()
        cleaned_category = re.sub(
            r"[^\w\s]", "", cleaned_category
        )  # Remove unwanted characters
        cleaned_category_lower = cleaned_category.lower()

        # Exact match (case-insensitive)
        if cleaned_category_lower in self.classes:
            return cleaned_category_lower

        # Use difflib.get_close_matches to find close matches
        closest_matches = get_close_matches(
            cleaned_category_lower, self.classes, n=1, cutoff=0.8
        )
        if closest_matches:
            return closest_matches[0]

        # Use SequenceMatcher to find the best match
        max_similarity = 0
        best_class = None
        for class_name in self.classes:
            similarity = SequenceMatcher(
                None, cleaned_category_lower, class_name
            ).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                best_class = class_name

        # Set a similarity threshold to avoid incorrect mappings
        similarity_threshold = 0.6
        if max_similarity >= similarity_threshold:
            return best_class
        return best_class

    def parse_model_output(self, generated_text):
        """
        Parses the model's generated output to extract the predicted class.

        Args:
            generated_text (str): The output text generated by the model.

        Returns:
            predicted_class (str or None): The predicted class, or None if parsing fails.
        """
        # The generated text is expected to be the category name
        if "llama-3" in self.model_id.lower():
            # Remove 'assistant' prefix and clean up whitespace/newlines
            text = generated_text.replace("assistant", "").strip()
            # Split on newlines and take the first non-empty line
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            category_text = lines[0] if lines else ""
        else:
            # For LLaMA-2, check for various instruction tag formats
            for tag in ["[/INST]", "INST]", "/INST]", "]"]:
                if tag in generated_text:
                    category_text = generated_text.split(tag)[-1].strip()
                    break
            else:
                category_text = generated_text.strip()

        # Map the generated category to a known class
        predicted_class = self.map_generated_category_to_class(category_text)
        # print("Predicted class:", predicted_class)

        return predicted_class
