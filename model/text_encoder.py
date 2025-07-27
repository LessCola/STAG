from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F


class TextEncoder:
    def __init__(self, model_name: str, device):
        """
        Initialize the AutoModel for Sentence Transformer-style embeddings using a model ID from the Hugging Face Hub.

        Args:
        model_name (str): The name of the model to load. ['sentence-transformers/paraphrase-MiniLM-L6-v2', etc.]
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()  # Set model to evaluation mode

    def embed(self, texts, output_format: str = "tensor"):
        """
        Returns the embedding of the input texts using the AutoModel representation.

        Args:
            texts (str or list of str): Input text or list of texts to embed.
            output_format (str): Format of the output, either "numpy" or "tensor".

        Returns:
            Embedding(s) in the specified format.
        """
        if isinstance(texts, str):
            texts = [texts]  # Convert single text to list

        # Tokenize the input texts
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Get the embeddings from the model
        with torch.no_grad():
            model_output = self.model(**inputs)

        # Mean pooling with attention mask to ignore padding tokens
        token_embeddings = model_output.last_hidden_state
        attention_mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        # Apply attention mask
        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        # L2 normalize the embeddings to replicate SentenceTransformer behavior
        normalized_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=1)

        if output_format == "numpy":
            return normalized_embeddings.cpu().numpy()
        elif output_format == "tensor":
            return normalized_embeddings
        else:
            raise ValueError(f"Unknown output format: {output_format}")
