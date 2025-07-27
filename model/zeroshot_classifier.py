import copy
import os
import re
from difflib import SequenceMatcher, get_close_matches

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT_TEMPLATE = {
    "zero-shot": [
        (
            f"You are an AI assistant tasked with classifying input word sequences into one of the following categories: {'{classes}'}.\n"
            "You must choose strictly from these categories and no others.\n"
        ),
        (
            "When given a new input sequence, classify it into one of the categories.\n"
            "**IMPORTANT:** Output only the category name and nothing else."
        ),
    ],
}


class ZeroShotClassifier:
    def __init__(
        self,
        model_name,
        device,
        max_length=4096,
    ):
        self.model_id = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).half()
        if "llama-3" in self.model_id.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=True, padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

        else:
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

    def create_zero_shot_prompt(self):
        """Creates the conversation prompt for zero-shot classification."""
        selected_classes = self.classes.copy()
        np.random.shuffle(
            selected_classes
        )  # Shuffle classes to avoid initial order bias
        prompt_template = PROMPT_TEMPLATE["zero-shot"]

        system_prompt = prompt_template[0].format(classes=", ".join(selected_classes))
        final_instruction = prompt_template[1]

        # Combine system prompt and final instruction (no examples)
        prompt = system_prompt + final_instruction
        # print("Prompt:\n", prompt)

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
        prompt = ""
        for turn in conversation:
            if turn["role"] == "system":
                prompt += f"<s>[INST] <<SYS>>\n{turn['content']}\n<</SYS>>\n[/INST]"
            elif turn["role"] == "user":
                prompt += f"<s>[INST] {turn['content']} [/INST]"
            elif turn["role"] == "assistant":
                prompt += f"{turn['content']}</s>"
        return prompt

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
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
                # attention_mask = torch.ones_like(input_ids)

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
