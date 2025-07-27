import copy
import logging
import re
import sys
from copy import deepcopy

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm

from src.lc_sampler import (
    LinearProbingDataLoader,
    setup_eval_dataloader,
    # setup_eval_small_dataloader,
)
from src.utils import accuracy, set_random_seed

terminal_stdout = getattr(sys.stdout, "terminal_stdout", sys.stdout)


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        self.init_parameters()

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits

    def init_parameters(self):
        # Initialize weights and biases
        init.kaiming_uniform_(
            self.linear.weight, a=0, mode="fan_in", nonlinearity="relu"
        )
        if self.linear.bias is not None:
            init.constant_(self.linear.bias, 0)


def collate_fn(batch):
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    # * Unsqueeze each label tensor to add a dimension
    labels = [label.unsqueeze(0) if label.dim() == 0 else label for label in labels]
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


def clean_class_name(cls_name):
    return re.sub(r"[^a-zA-Z0-9_\s]+", "", cls_name.lower().replace("_", " "))


def print_examples_from_list(data):
    # Initialize an empty dictionary to store categories and their examples
    category_dict = {}

    # Process each item in the list
    for item in data:
        # Extract input sequence and category using string formatting
        try:
            parts = item.split("Category:")
            input_sequence = parts[0].replace("Input:", "").strip()
            category = parts[1].strip()

            # Add input sequence to the category in the dictionary
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(input_sequence)
        except IndexError:
            continue  # In case the string does not follow the expected format

    # Sort categories alphabetically
    sorted_categories = dict(sorted(category_dict.items()))

    # Print the sorted dictionary
    for category, input_sequences in sorted_categories.items():
        print(f"# Category: {category}")
        for sequence in input_sequences:
            print(f"  Input: {sequence}")


def print_examples(data):
    if isinstance(data[0], str):
        print_examples_from_list(data)
        return
    # Initialize an empty dictionary to store categories and their examples
    category_dict = {}

    # Regular expression pattern to find input sequences and their categories
    pattern = r"- Input: (.*?)\n\s*Category:\s*(.*)"

    # Process the input data
    for item in data:
        content = item["content"]

        # Find all matches of input sequences and their categories
        matches = re.findall(pattern, content)

        # Add matches to the category dictionary
        for input_sequence, category in matches:
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(input_sequence.strip())

    # Sort categories alphabetically
    sorted_categories = dict(sorted(category_dict.items()))

    # Print the sorted dictionary
    for category, input_sequences in sorted_categories.items():
        print(f"# Category: {category}")
        for sequence in input_sequences:
            print(f"  Input: {sequence}")


def fewshot_with_llm(
    model,
    classifier,
    graph,
    feats,
    nodes,
    categories,
    way,
    shot,
    batch_size,
    device,
):
    """
    Run inference once but test on multiple tasks using different masks.

    Args:
        categories: List of categories
        few_shot_train_masks: Training masks of shape (num_nodes, num_tasks)
        few_shot_val_masks: Validation masks of shape (num_nodes, num_tasks)
        few_shot_test_masks: Test masks of shape (num_nodes, num_tasks)
    """

    def process_batches(tokens, indices, true_categories, set_name, batch_size):
        """Helper function to process batches and calculate accuracy."""
        pred = []
        correct_predictions = 0

        # Retrieve token batches
        token_batches = [tokens[idx] for idx in indices]

        # Initialize the progress bar
        progress_bar = tqdm(
            range(0, len(token_batches), batch_size),
            desc=f"Processing {set_name} batches",
            file=terminal_stdout,
        )
        total_samples = 0
        # Batch processing loop
        for i in progress_bar:
            batch_tokens = token_batches[i : i + batch_size]
            batch_pred = classifier.classify(batch_tokens)
            pred.extend(batch_pred)
            total_samples += len(batch_pred)

            # Update correct predictions
            correct_predictions += np.sum(
                np.array(batch_pred) == true_categories[i : i + batch_size]
            )

            # Calculate and update accuracy in progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_description(
                f"Processing {set_name} batches (Acc: {current_acc:.4f})"
            )

        # Final accuracy calculation
        final_acc = np.mean(np.array(pred) == true_categories)
        print(f"--- {set_name.capitalize()} Acc: {final_acc:.4f} ---")
        return final_acc

    # ! create few-shot masks with fixed query size
    graph, selected_classes_list = create_few_shot_multitask_masks_fixed_query_size(
        graph, categories, way, shot, include_val=True
    )
    few_shot_train_masks, few_shot_val_masks, few_shot_test_masks = (
        graph.ndata["few_shot_train_mask"],
        graph.ndata["few_shot_val_mask"],
        graph.ndata["few_shot_test_mask"],
    )

    eval_loader = setup_eval_dataloader(
        "lc",
        graph,
        feats,
        nodes,
        64,  # 512
    )

    # Run inference once
    with torch.no_grad():
        model.eval()
        embeddings = []
        tokens = []

        for batch in tqdm(eval_loader, desc="Infering...", file=terminal_stdout):
            batch_g, targets, _, node_idx = batch
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            targets = targets.to(device)
            batch_emb, batch_tokens = model.embed_quant(batch_g, x)
            batch_emb = batch_emb[targets]
            batch_tokens = [batch_tokens[i] for i in targets]
            embeddings.append(batch_emb.cpu())
            tokens.extend(batch_tokens)
    embeddings = torch.cat(embeddings, dim=0)

    # Get number of tasks from mask shape
    num_tasks = few_shot_train_masks.shape[1]

    # Process each task
    test_accs = []
    val_accs = []
    categories = np.array([clean_class_name(cls) for cls in categories])
    for task_idx in range(num_tasks):
        print(f"\nProcessing Task {task_idx}")

        # Get masks for current task
        selected_classes = selected_classes_list[task_idx]
        train_mask = few_shot_train_masks[:, task_idx]
        val_mask = few_shot_val_masks[:, task_idx]
        test_mask = few_shot_test_masks[:, task_idx]

        # Create few-shot prompt for current task
        classifier.set_classes(selected_classes)
        classifier.create_few_shot_prompt(train_mask, tokens, categories)
        # print_examples(classifier.conversation_template)

        # Process validation set
        val_indices = torch.nonzero(val_mask, as_tuple=True)[0].numpy()
        val_categories = categories[val_mask]
        val_acc = process_batches(
            tokens, val_indices, val_categories, f"Task {task_idx} Val Set", batch_size
        )
        val_accs.append(val_acc)

        # Process test set
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0].numpy()
        test_categories = categories[test_mask]
        test_acc = process_batches(
            tokens,
            test_indices,
            test_categories,
            f"Task {task_idx} Test Set",
            batch_size,
        )
        test_accs.append(test_acc)
        print(
            f"--- Average Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f} ---"
        )
        print(
            f"--- Average Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f} ---"
        )

    return test_accs, val_accs


def weighted_contrastive_loss(
    cosine_similarities, true_labels, class_similarity_weights, temperature=0.3
):
    """
    Weighted contrastive loss that properly considers class similarity weights.

    Args:
        cosine_similarities (torch.Tensor): Tensor of shape (batch_size, num_classes)
        true_labels (torch.Tensor): Tensor of shape (batch_size,)
        class_similarity_weights (torch.Tensor): Tensor of shape (num_classes, num_classes)
        temperature (float): Temperature parameter to scale the logits

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Get weights for true class pairs
    batch_weights = class_similarity_weights[
        true_labels
    ]  # Shape: (batch_size, num_classes)

    # Apply weights to scaled similarities
    weighted_similarities = batch_weights * cosine_similarities / temperature

    # Compute log softmax on weighted similarities
    log_probs = F.log_softmax(weighted_similarities, dim=1)

    # Compute cross entropy with one-hot targets
    target_distribution = F.one_hot(
        true_labels, num_classes=cosine_similarities.size(-1)
    ).float()

    loss = -torch.sum(target_distribution * log_probs, dim=1).mean()

    return loss


def fewshot_prompt_tuning_with_llm(
    model,
    classifier,
    graph,
    feats,
    nodes,
    labels,
    categories,
    way,
    shot,
    batch_size,
    max_epochs_f,
    lr_f,
    weight_decay_f,
    batch_size_f,
    device,
):
    """
    Run inference once but test on multiple tasks using different masks.

    Args:
        categories: List of categories
        few_shot_train_masks: Training masks of shape (num_nodes, num_tasks)
        few_shot_val_masks: Validation masks of shape (num_nodes, num_tasks)
        few_shot_test_masks: Test masks of shape (num_nodes, num_tasks)
    """

    def process_batches(tokens, indices, true_categories, set_name, batch_size):
        """Helper function to process batches and calculate accuracy."""
        pred = []
        correct_predictions = 0

        # Retrieve token batches
        token_batches = [tokens[idx] for idx in indices]

        # Initialize the progress bar
        progress_bar = tqdm(
            range(0, len(token_batches), batch_size),
            desc=f"Processing {set_name} batches",
            file=terminal_stdout,
        )
        total_samples = 0
        # Batch processing loop
        for i in progress_bar:
            batch_tokens = token_batches[i : i + batch_size]
            batch_pred = classifier.classify(batch_tokens)
            pred.extend(batch_pred)
            total_samples += len(batch_pred)

            # Update correct predictions
            correct_predictions += np.sum(
                np.array(batch_pred) == true_categories[i : i + batch_size]
            )

            # Calculate and update accuracy in progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_description(
                f"Processing {set_name} batches (Acc: {current_acc:.4f})"
            )

        # Final accuracy calculation
        final_acc = np.mean(np.array(pred) == true_categories)
        print(f"--- {set_name.capitalize()} Acc: {final_acc:.4f} ---")
        return final_acc

    # create few-shot masks with fixed query size
    graph, selected_classes_list = create_few_shot_multitask_masks_fixed_query_size(
        graph, categories, way, shot, include_val=True
    )
    few_shot_train_masks, few_shot_val_masks, few_shot_test_masks = (
        graph.ndata["few_shot_train_mask"],
        graph.ndata["few_shot_val_mask"],
        graph.ndata["few_shot_test_mask"],
    )

    eval_loader = setup_eval_dataloader(
        "lc",
        graph,
        feats,
        nodes,
        64,
    )

    # Run inference once
    with torch.no_grad():
        model.eval()
        embeddings = []

        for batch in tqdm(eval_loader, desc="Infering...", file=terminal_stdout):
            batch_g, targets, _, node_idx = batch
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            targets = targets.to(device)
            batch_emb = model.embed(batch_g, x)
            batch_emb = batch_emb[targets]
            embeddings.append(batch_emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    # Get number of tasks from mask shape
    num_tasks = few_shot_train_masks.shape[1]

    # Process each task
    test_accs = []
    val_accs = []
    categories = np.array([clean_class_name(cls) for cls in categories])
    for task_idx in range(num_tasks):
        print(f"\nProcessing Task {task_idx}")
        prompt_quantizer = deepcopy(model.prompt_quantizer)
        prompt_quantizer.update_codebook(few_shot_train_masks[:, task_idx])
        label_to_idx = prompt_quantizer.label_to_idx
        label_map = torch.tensor(list(label_to_idx.values())).to(device)
        label_keys = torch.tensor(list(label_to_idx.keys())).to(device)

        # Get masks for current task
        selected_classes = selected_classes_list[task_idx]
        train_mask = few_shot_train_masks[:, task_idx]
        val_mask = few_shot_val_masks[:, task_idx]
        test_mask = few_shot_test_masks[:, task_idx]

        train_indices = torch.nonzero(train_mask, as_tuple=True)[0].numpy()
        train_nodes = [nodes[i] for i in train_indices]

        # setup_eval_small_dataloader
        train_loader = setup_eval_dataloader(
            "lc",
            graph,
            embeddings,  # * use the encoded embeddings
            train_nodes,
            batch_size_f,
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(
            prompt_quantizer.parameters(), lr=lr_f, weight_decay=weight_decay_f
        )
        progress_bar = tqdm(
            range(max_epochs_f), desc="Training...", file=terminal_stdout
        )
        model.eval()
        prompt_quantizer.train()
        for epoch in progress_bar:
            epoch_loss = 0
            epoch_prompting_loss = 0
            for batch in train_loader:
                batch_g, targets, _, node_idx = batch
                batch_g = batch_g.to(device)
                batch_emb = batch_g.ndata["feat"]
                targets = targets.to(device)
                batch_emb = batch_emb[targets]
                batch_labels = batch_g.ndata["label"][targets]
                batch_labels = label_map[
                    torch.where(label_keys == batch_labels.unsqueeze(1))[1]
                ]
                loss, z_q, tokens, cosine_similarity = prompt_quantizer(batch_emb)
                prompting_loss = weighted_contrastive_loss(
                    cosine_similarity,
                    batch_labels,
                    prompt_quantizer.class_similarity_weights,
                    prompt_quantizer.tau_f,
                )
                loss += prompting_loss
                epoch_prompting_loss += prompting_loss.item()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss /= len(train_loader)
            epoch_prompting_loss /= len(train_loader)
            progress_bar.set_description(
                f"Epoch {epoch} Loss: {epoch_loss:.4f}, Prompting Loss: {epoch_prompting_loss:.4f}"
            )

        val_indices = torch.nonzero(val_mask, as_tuple=True)[0].numpy()
        val_nodes = [nodes[i] for i in val_indices]
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0].numpy()
        test_nodes = [nodes[i] for i in test_indices]
        task_nodes = train_nodes + val_nodes + test_nodes
        task_loader = setup_eval_dataloader(
            "lc",
            graph,
            embeddings,  # * use the encoded embeddings
            task_nodes,
            64,
            shuffle=False,
        )
        task_tokens = []

        with torch.no_grad():
            model.eval()
            prompt_quantizer.eval()
            for batch in tqdm(task_loader, desc="Quantizing...", file=terminal_stdout):
                batch_g, targets, _, node_idx = batch
                batch_g = batch_g.to(device)
                x = batch_g.ndata.pop("feat")
                targets = targets.to(device)
                # ! batch_emb: embeddings of ego nodes
                batch_emb, batch_tokens = model.prompt_quant(
                    x, prompt_quantizer.cond_net(x)
                )
                batch_emb = batch_emb[targets]
                batch_tokens = [batch_tokens[i] for i in targets]
                task_tokens.extend(batch_tokens)

        tokens = [None] * len(graph.nodes())
        for i in train_mask.nonzero(as_tuple=True)[0]:
            tokens[i] = task_tokens.pop(0)
        for i in val_mask.nonzero(as_tuple=True)[0]:
            tokens[i] = task_tokens.pop(0)
        for i in test_mask.nonzero(as_tuple=True)[0]:
            tokens[i] = task_tokens.pop(0)

        # Create few-shot prompt for current task
        classifier.set_classes(selected_classes)
        classifier.create_few_shot_prompt(train_mask, tokens, categories)
        # print_examples(classifier.conversation_template)

        # Process validation set
        val_indices = torch.nonzero(val_mask, as_tuple=True)[0].numpy()
        val_categories = categories[val_mask]
        val_acc = process_batches(
            tokens, val_indices, val_categories, f"Task {task_idx} Val Set", batch_size
        )
        val_accs.append(val_acc)

        # Process test set
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0].numpy()
        test_categories = categories[test_mask]
        test_acc = process_batches(
            tokens,
            test_indices,
            test_categories,
            f"Task {task_idx} Test Set",
            batch_size,
        )
        test_accs.append(test_acc)
        print(
            f"--- Average Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f} ---"
        )
        print(
            f"--- Average Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f} ---"
        )

    return test_accs, val_accs


def fewshot_prompt_tuning_without_llm(
    model,
    graph,
    feats,
    nodes,
    labels,
    categories,
    way,
    shot,
    batch_size,
    max_epochs_f,
    lr_f,
    weight_decay_f,
    batch_size_f,
    device,
):
    """
    Run inference once but test on multiple tasks using different masks.

    Args:
        categories: List of categories
        few_shot_train_masks: Training masks of shape (num_nodes, num_tasks)
        few_shot_val_masks: Validation masks of shape (num_nodes, num_tasks)
        few_shot_test_masks: Test masks of shape (num_nodes, num_tasks)
    """

    def process_batches(tokens, indices, true_categories, set_name, batch_size):
        """Helper function to process batches and calculate accuracy."""
        pred = []
        correct_predictions = 0

        # Retrieve token batches
        token_batches = [tokens[idx] for idx in indices]

        # Initialize the progress bar
        progress_bar = tqdm(
            range(0, len(token_batches), batch_size),
            desc=f"Processing {set_name} batches",
            file=terminal_stdout,
        )
        total_samples = 0
        # Batch processing loop
        for i in progress_bar:
            batch_tokens = token_batches[i : i + batch_size]
            batch_pred = batch_tokens
            pred.extend(batch_pred)
            total_samples += len(batch_pred)

            # Update correct predictions
            correct_predictions += np.sum(
                np.array(batch_pred) == true_categories[i : i + batch_size]
            )

            # Calculate and update accuracy in progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_description(
                f"Processing {set_name} batches (Acc: {current_acc:.4f})"
            )

        # Final accuracy calculation
        final_acc = np.mean(np.array(pred) == true_categories)
        print(f"--- {set_name.capitalize()} Acc: {final_acc:.4f} ---")
        return final_acc

    # * create few-shot masks with fixed query size
    graph, selected_classes_list = create_few_shot_multitask_masks_fixed_query_size(
        graph,
        categories,
        way,
        shot,
        include_val=True,
    )
    few_shot_train_masks, few_shot_val_masks, few_shot_test_masks = (
        graph.ndata["few_shot_train_mask"],
        graph.ndata["few_shot_val_mask"],
        graph.ndata["few_shot_test_mask"],
    )

    eval_loader = setup_eval_dataloader(
        "lc",
        graph,
        feats,
        nodes,
        64,  # 512
    )

    # Run inference once
    with torch.no_grad():
        model.eval()
        embeddings = []

        for batch in tqdm(eval_loader, desc="Infering...", file=terminal_stdout):
            batch_g, targets, _, node_idx = batch
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            targets = targets.to(device)
            batch_emb = model.embed(batch_g, x)
            batch_emb = batch_emb[targets]
            embeddings.append(batch_emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    # Get number of tasks from mask shape
    num_tasks = few_shot_train_masks.shape[1]

    # Process each task
    test_accs = []
    val_accs = []
    categories = np.array([clean_class_name(cls) for cls in categories])
    for task_idx in range(num_tasks):
        print(f"\nProcessing Task {task_idx}")
        prompt_quantizer = deepcopy(model.prompt_quantizer)
        prompt_quantizer.update_codebook(few_shot_train_masks[:, task_idx])
        label_to_idx = prompt_quantizer.label_to_idx
        label_map = torch.tensor(list(label_to_idx.values())).to(device)
        label_keys = torch.tensor(list(label_to_idx.keys())).to(device)
        # Get masks for current task
        # selected_classes = selected_classes_list[task_idx]
        train_mask = few_shot_train_masks[:, task_idx]
        val_mask = few_shot_val_masks[:, task_idx]
        test_mask = few_shot_test_masks[:, task_idx]

        train_indices = torch.nonzero(train_mask, as_tuple=True)[0].numpy()
        train_nodes = [nodes[i] for i in train_indices]

        # setup_eval_small_dataloader
        train_loader = setup_eval_dataloader(
            "lc",
            graph,
            embeddings,  # * use the encoded embeddings
            train_nodes,
            batch_size_f,
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(
            prompt_quantizer.parameters(), lr=lr_f, weight_decay=weight_decay_f
        )
        progress_bar = tqdm(
            range(max_epochs_f), desc="Training...", file=terminal_stdout
        )
        model.eval()
        prompt_quantizer.train()
        for epoch in progress_bar:
            epoch_loss = 0
            epoch_prompting_loss = 0
            for batch in train_loader:
                batch_g, targets, _, node_idx = batch
                batch_g = batch_g.to(device)
                # batch_emb = batch_g.ndata.pop("feat")
                batch_emb = batch_g.ndata["feat"]
                targets = targets.to(device)
                batch_emb = batch_emb[targets]
                batch_labels = batch_g.ndata["label"][targets]
                batch_labels = label_map[
                    torch.where(label_keys == batch_labels.unsqueeze(1))[1]
                ]
                loss, z_q, tokens, cosine_similarity = prompt_quantizer(batch_emb)
                prompting_loss = weighted_contrastive_loss(
                    cosine_similarity,
                    batch_labels,
                    prompt_quantizer.class_similarity_weights,
                    prompt_quantizer.tau_f,
                )
                loss += prompting_loss
                epoch_prompting_loss += prompting_loss.item()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss /= len(train_loader)
            epoch_prompting_loss /= len(train_loader)
            progress_bar.set_description(
                f"Epoch {epoch} Loss: {epoch_loss:.4f}, Prompting Loss: {epoch_prompting_loss:.4f}"
            )

        val_indices = torch.nonzero(val_mask, as_tuple=True)[0].numpy()
        val_nodes = [nodes[i] for i in val_indices]
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0].numpy()
        test_nodes = [nodes[i] for i in test_indices]
        task_nodes = train_nodes + val_nodes + test_nodes
        task_loader = setup_eval_dataloader(
            "lc",
            graph,
            embeddings,  # * use the encoded embeddings
            task_nodes,
            64,  # 512
            shuffle=False,
        )
        task_tokens = []

        with torch.no_grad():
            model.eval()
            prompt_quantizer.eval()
            for batch in tqdm(task_loader, desc="Quantizing...", file=terminal_stdout):
                batch_g, targets, _, node_idx = batch
                batch_g = batch_g.to(device)
                x = batch_g.ndata.pop("feat")
                targets = targets.to(device)
                x = x[targets]
                batch_emb, batch_tokens = prompt_quantizer(x, True)
                task_tokens.extend([t[0] for t in batch_tokens])

        tokens = [None] * len(graph.nodes())
        for i in train_mask.nonzero(as_tuple=True)[0]:
            tokens[i] = task_tokens.pop(0)
        for i in val_mask.nonzero(as_tuple=True)[0]:
            tokens[i] = task_tokens.pop(0)
        for i in test_mask.nonzero(as_tuple=True)[0]:
            tokens[i] = task_tokens.pop(0)

        # Process validation set
        val_indices = torch.nonzero(val_mask, as_tuple=True)[0].numpy()
        val_categories = categories[val_mask]
        val_acc = process_batches(
            tokens, val_indices, val_categories, f"Task {task_idx} Val Set", batch_size
        )
        val_accs.append(val_acc)

        # Process test set
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0].numpy()
        test_categories = categories[test_mask]
        test_acc = process_batches(
            tokens,
            test_indices,
            test_categories,
            f"Task {task_idx} Test Set",
            batch_size,
        )
        test_accs.append(test_acc)
        print(
            f"--- Average Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f} ---"
        )
        print(
            f"--- Average Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f} ---"
        )

    return test_accs, val_accs


def few_shot_linear_probing(
    model,
    graph,
    feats,
    labels,
    categories,
    nodes,
    n_way,
    k_shot,
    lr_f,
    weight_decay_f,
    max_epoch_f,
    batch_size_f,
    device,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
):
    logging.info("-- Linear Probing in downstream tasks ---")
    eval_loader = setup_eval_dataloader(
        "lc",
        graph,
        feats,
        nodes,
        512,
    )

    with torch.no_grad():
        model.eval()
        embeddings = []

        for batch in tqdm(eval_loader, desc="Infering...", file=terminal_stdout):
            batch_g, targets, _, node_idx = batch
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            targets = targets.to(device)
            # ! batch_emb: embeddings of ego nodes
            batch_emb = model.embed(batch_g, x)[targets]
            embeddings.append(batch_emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    set_random_seed(0)
    graph, selected_classes_list = create_few_shot_multitask_masks_fixed_query_size(
        graph,
        categories,
        n_way,
        k_shot,
        include_val=True,
    )
    num_tasks = len(selected_classes_list)
    test_acc_list = []
    best_val_acc_list = []
    for task_idx in range(num_tasks):
        print(f"Task {task_idx} of {num_tasks}")
        train_mask, val_mask, test_mask = (
            graph.ndata["few_shot_train_mask"][:, task_idx],
            graph.ndata["few_shot_val_mask"][:, task_idx],
            graph.ndata["few_shot_test_mask"][:, task_idx],
        )
        train_emb, val_emb, test_emb = (
            embeddings[train_mask],
            embeddings[val_mask],
            embeddings[test_mask],
        )
        train_lbls, val_lbls, test_lbls = (
            labels[train_mask],
            labels[val_mask],
            labels[test_mask],
        )
        # if not tuning:
        test_acc, best_val_acc = node_classification_linear_probing(
            (train_emb, val_emb, test_emb),
            (train_lbls, val_lbls, test_lbls),
            lr_f,
            weight_decay_f,
            max_epoch_f,
            batch_size_f,
            device,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        test_acc_list.append(test_acc)
        best_val_acc_list.append(best_val_acc)
        print(
            f"--- Average Val acc: {np.mean(best_val_acc_list):.4f}±{np.std(best_val_acc_list):.4f} ---"
        )
        print(
            f"--- Average Test acc: {np.mean(test_acc_list):.4f}±{np.std(test_acc_list):.4f} ---"
        )

    # return np.mean(test_acc_list), np.mean(best_val_acc_list)
    return test_acc_list, best_val_acc_list


def node_classification_linear_probing(
    embeddings,
    labels,
    lr,
    weight_decay,
    max_epoch,
    batch_size,
    device,
    shuffle=True,
    mute=False,
    # * default values
    num_workers=4,
    persistent_workers=True,
):
    criterion = torch.nn.CrossEntropyLoss()

    train_emb, val_emb, test_emb = embeddings
    train_label, val_label, test_label = labels
    train_label = train_label.to(torch.long)
    val_label = val_label.to(torch.long)
    test_label = test_label.to(torch.long)

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch), file=terminal_stdout)
    else:
        epoch_iter = range(max_epoch)

    encoder = LogisticRegression(train_emb.shape[1], int(train_label.max().item() + 1))
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    if batch_size > 0:
        train_loader = LinearProbingDataLoader(
            np.arange(len(train_emb)),
            train_emb,
            train_label,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            shuffle=shuffle,
            worker_init_fn=lambda worker_id: np.random.seed(worker_id),
        )
        val_loader = LinearProbingDataLoader(
            np.arange(len(val_emb)),
            val_emb,
            val_label,
            batch_size=batch_size,
            # num_workers=4,
            # persistent_workers=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            shuffle=False,
            worker_init_fn=lambda worker_id: np.random.seed(worker_id),
        )
        test_loader = LinearProbingDataLoader(
            np.arange(len(test_emb)),
            test_emb,
            test_label,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            shuffle=False,
            worker_init_fn=lambda worker_id: np.random.seed(worker_id),
        )
    else:
        train_loader = [np.arange(len(train_emb))]
        val_loader = [np.arange(len(val_emb))]
        test_loader = [np.arange(len(test_emb))]

    def eval_forward(loader, _label):
        pred_all = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            pred = encoder(None, batch_x)
            pred_all.append(pred.cpu())
        pred = torch.cat(pred_all, dim=0)
        acc = accuracy(pred, _label)
        return acc

    for epoch in epoch_iter:
        encoder.train()

        for batch_x, batch_label in train_loader:
            batch_x = batch_x.to(device)
            batch_label = batch_label.to(device)
            pred = encoder(None, batch_x)
            loss = criterion(pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            encoder.eval()
            val_acc = eval_forward(val_loader, val_label)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(encoder)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc:.4f}"
            )

    best_model.eval()
    encoder = best_model
    with torch.no_grad():
        test_acc = eval_forward(test_loader, test_label)
    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- "
        )
    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- "
        )
    # * clean up
    train_loader = None
    val_loader = None
    test_loader = None

    return test_acc, best_val_acc


def zeroshot_with_llm(
    model,
    classifier,
    graph,
    feats,
    nodes,
    categories,
    way,
    batch_size,
    device,
):
    """
    Run inference once but test on multiple tasks using different masks.

    Args:
        categories: List of categories
        few_shot_train_masks: Training masks of shape (num_nodes, num_tasks)
        few_shot_val_masks: Validation masks of shape (num_nodes, num_tasks)
        few_shot_test_masks: Test masks of shape (num_nodes, num_tasks)
    """

    def process_batches(tokens, indices, true_categories, set_name, batch_size):
        """Helper function to process batches and calculate accuracy."""
        pred = []
        correct_predictions = 0

        # Retrieve token batches
        token_batches = [tokens[idx] for idx in indices]

        # Initialize the progress bar
        progress_bar = tqdm(
            range(0, len(token_batches), batch_size),
            desc=f"Processing {set_name} batches",
            file=terminal_stdout,
        )
        total_samples = 0
        # Batch processing loop
        for i in progress_bar:
            batch_tokens = token_batches[i : i + batch_size]
            batch_pred = classifier.classify(batch_tokens)
            pred.extend(batch_pred)
            total_samples += len(batch_pred)

            # Update correct predictions
            correct_predictions += np.sum(
                np.array(batch_pred) == true_categories[i : i + batch_size]
            )

            # Calculate and update accuracy in progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_description(
                f"Processing {set_name} batches (Acc: {current_acc:.4f})"
            )

        # Final accuracy calculation
        final_acc = np.mean(np.array(pred) == true_categories)
        print(f"--- {set_name.capitalize()} Acc: {final_acc:.4f} ---")
        return final_acc

    # ! create few-shot masks with fixed query size
    graph, selected_classes_list = create_few_shot_multitask_masks_fixed_query_size(
        graph, categories, way, 5, include_val=True
    )
    few_shot_train_masks, few_shot_val_masks, few_shot_test_masks = (
        graph.ndata["few_shot_train_mask"],
        graph.ndata["few_shot_val_mask"],
        graph.ndata["few_shot_test_mask"],
    )

    eval_loader = setup_eval_dataloader(
        "lc",
        graph,
        feats,
        nodes,
        64,  # 512
    )

    # Run inference once
    with torch.no_grad():
        model.eval()
        embeddings = []
        tokens = []

        for batch in tqdm(eval_loader, desc="Infering...", file=terminal_stdout):
            batch_g, targets, _, node_idx = batch
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            targets = targets.to(device)
            batch_emb, batch_tokens = model.embed_quant(batch_g, x)
            batch_emb = batch_emb[targets]
            batch_tokens = [batch_tokens[i] for i in targets]
            embeddings.append(batch_emb.cpu())
            tokens.extend(batch_tokens)
    embeddings = torch.cat(embeddings, dim=0)

    # Get number of tasks from mask shape
    num_tasks = few_shot_train_masks.shape[1]

    # Process each task
    test_accs = []
    val_accs = []
    categories = np.array([clean_class_name(cls) for cls in categories])
    for task_idx in range(num_tasks):
        print(f"\nProcessing Task {task_idx}")

        # Get masks for current task
        selected_classes = selected_classes_list[task_idx]
        val_mask = few_shot_val_masks[:, task_idx]
        test_mask = few_shot_test_masks[:, task_idx]

        # Create few-shot prompt for current task
        classifier.set_classes(selected_classes)
        classifier.create_zero_shot_prompt()

        # Process validation set
        val_indices = torch.nonzero(val_mask, as_tuple=True)[0].numpy()
        val_categories = categories[val_mask]
        val_acc = process_batches(
            tokens, val_indices, val_categories, f"Task {task_idx} Val Set", batch_size
        )
        val_accs.append(val_acc)

        # Process test set
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0].numpy()
        test_categories = categories[test_mask]
        test_acc = process_batches(
            tokens,
            test_indices,
            test_categories,
            f"Task {task_idx} Test Set",
            batch_size,
        )
        test_accs.append(test_acc)
        print(
            f"--- Average Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f} ---"
        )
        print(
            f"--- Average Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f} ---"
        )

    return test_accs, val_accs


def zeroshot_without_llm(
    model,
    graph,
    feats,
    nodes,
    labels,
    categories,
    way,
    batch_size,
    device,
):
    """
    Run inference once but test on multiple tasks using different masks.

    Args:
        categories: List of categories
        few_shot_train_masks: Training masks of shape (num_nodes, num_tasks)
        few_shot_val_masks: Validation masks of shape (num_nodes, num_tasks)
        few_shot_test_masks: Test masks of shape (num_nodes, num_tasks)
    """

    def process_batches(tokens, indices, true_categories, set_name, batch_size):
        """Helper function to process batches and calculate accuracy."""
        pred = []
        correct_predictions = 0

        # Retrieve token batches
        token_batches = [tokens[idx] for idx in indices]

        # Initialize the progress bar
        progress_bar = tqdm(
            range(0, len(token_batches), batch_size),
            desc=f"Processing {set_name} batches",
            file=terminal_stdout,
        )
        total_samples = 0
        # Batch processing loop
        for i in progress_bar:
            batch_tokens = token_batches[i : i + batch_size]
            batch_pred = batch_tokens
            pred.extend(batch_pred)
            total_samples += len(batch_pred)

            # Update correct predictions
            correct_predictions += np.sum(
                np.array(batch_pred) == true_categories[i : i + batch_size]
            )

            # Calculate and update accuracy in progress bar
            current_acc = correct_predictions / total_samples
            progress_bar.set_description(
                f"Processing {set_name} batches (Acc: {current_acc:.4f})"
            )

        # Final accuracy calculation
        final_acc = np.mean(np.array(pred) == true_categories)
        print(f"--- {set_name.capitalize()} Acc: {final_acc:.4f} ---")
        return final_acc

    # * create few-shot masks with fixed query size
    graph, selected_classes_list = create_few_shot_multitask_masks_fixed_query_size(
        graph,
        categories,
        way,
        5,
        include_val=True,
    )
    few_shot_train_masks, few_shot_val_masks, few_shot_test_masks = (
        graph.ndata["few_shot_train_mask"],
        graph.ndata["few_shot_val_mask"],
        graph.ndata["few_shot_test_mask"],
    )

    eval_loader = setup_eval_dataloader(
        "lc",
        graph,
        feats,
        nodes,
        64,  # 512
    )

    # Run inference once
    with torch.no_grad():
        model.eval()
        embeddings = []

        for batch in tqdm(eval_loader, desc="Infering...", file=terminal_stdout):
            batch_g, targets, _, node_idx = batch
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            targets = targets.to(device)
            batch_emb = model.embed(batch_g, x)
            batch_emb = batch_emb[targets]
            embeddings.append(batch_emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    # Get number of tasks from mask shape
    num_tasks = few_shot_train_masks.shape[1]

    # Process each task
    test_accs = []
    val_accs = []
    categories = np.array([clean_class_name(cls) for cls in categories])
    for task_idx in range(num_tasks):
        print(f"\nProcessing Task {task_idx}")
        prompt_quantizer = deepcopy(model.prompt_quantizer)
        prompt_quantizer.update_codebook(few_shot_train_masks[:, task_idx])
        train_mask = few_shot_train_masks[:, task_idx]
        val_mask = few_shot_val_masks[:, task_idx]
        test_mask = few_shot_test_masks[:, task_idx]

        train_indices = torch.nonzero(train_mask, as_tuple=True)[0].numpy()
        train_nodes = [nodes[i] for i in train_indices]

        val_indices = torch.nonzero(val_mask, as_tuple=True)[0].numpy()
        val_nodes = [nodes[i] for i in val_indices]
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0].numpy()
        test_nodes = [nodes[i] for i in test_indices]
        task_nodes = train_nodes + val_nodes + test_nodes
        task_loader = setup_eval_dataloader(
            "lc",
            graph,
            embeddings,  # * use the encoded embeddings
            task_nodes,
            64,  # 512
            shuffle=False,
        )
        task_tokens = []

        with torch.no_grad():
            model.eval()
            prompt_quantizer.eval()
            for batch in tqdm(task_loader, desc="Quantizing...", file=terminal_stdout):
                batch_g, targets, _, node_idx = batch
                batch_g = batch_g.to(device)
                x = batch_g.ndata.pop("feat")
                targets = targets.to(device)
                x = x[targets]
                batch_emb, batch_tokens, _ = prompt_quantizer.quantize_torch(
                    x, prompt=False
                )
                task_tokens.extend([t[0] for t in batch_tokens])

        tokens = [None] * len(graph.nodes())
        for i in train_mask.nonzero(as_tuple=True)[0]:
            tokens[i] = task_tokens.pop(0)
        for i in val_mask.nonzero(as_tuple=True)[0]:
            tokens[i] = task_tokens.pop(0)
        for i in test_mask.nonzero(as_tuple=True)[0]:
            tokens[i] = task_tokens.pop(0)

        # Process validation set
        val_indices = torch.nonzero(val_mask, as_tuple=True)[0].numpy()
        val_categories = categories[val_mask]
        val_acc = process_batches(
            tokens, val_indices, val_categories, f"Task {task_idx} Val Set", batch_size
        )
        val_accs.append(val_acc)

        # if not tuning:
        # Process test set
        test_indices = torch.nonzero(test_mask, as_tuple=True)[0].numpy()
        test_categories = categories[test_mask]
        test_acc = process_batches(
            tokens,
            test_indices,
            test_categories,
            f"Task {task_idx} Test Set",
            batch_size,
        )
        test_accs.append(test_acc)
        print(
            f"--- Average Val Acc: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f} ---"
        )
        # if not tuning:
        print(
            f"--- Average Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f} ---"
        )

    return test_accs, val_accs


def create_few_shot_multitask_masks_fixed_query_size(
    graph, categories, N, K, query_size=2000, num_tasks=20, seed=42, include_val=False
):
    """
    Generate masks for multiple few-shot tasks with train samples and evenly distributed query size.

    Args:
        graph: DGL graph
        categories: Category names
        N: Number of classes per task (N-way)
        K: Number of samples per class for the train set (K-shot)
        query_size: Total query size across all tasks
        num_tasks: Number of tasks to generate
        seed: Random seed

    Returns:
        graph: Graph with added multitask masks
        selected_classes_list: List of selected classes for each task
    """
    local_random = np.random.RandomState(seed)
    labels = graph.ndata["label"]

    # Initialize multitask masks
    few_shot_train_masks = torch.zeros((num_tasks, len(labels)), dtype=torch.bool)
    few_shot_val_masks = torch.zeros((num_tasks, len(labels)), dtype=torch.bool)
    few_shot_test_masks = torch.zeros((num_tasks, len(labels)), dtype=torch.bool)

    # Filter valid classes: at least K+1 samples and not in "unknown"/"nan"
    valid_classes = []
    unique_classes = torch.unique(labels).numpy()
    N = min(N, len(unique_classes))

    for cls in unique_classes:
        class_indices = (labels == cls).nonzero(as_tuple=True)[0]
        cls_name = categories[class_indices[0]]
        if len(class_indices) >= (include_val + 1) * K + 1 and cls_name.lower() not in [
            "unknown",
            "nan",
        ]:
            valid_classes.append(cls)

    if len(valid_classes) < N:
        raise ValueError(
            f"Not enough valid classes ({len(valid_classes)}) to select {N} classes"
        )

    selected_classes_list = []
    remaining_queries = query_size

    # Generate masks for each task
    for task in range(num_tasks):
        # Randomly select N classes for this task
        selected_classes = local_random.choice(valid_classes, size=N, replace=False)

        # Determine query size per class for this task
        base_Q = (remaining_queries // (num_tasks - task)) // N
        remaining_per_task = (remaining_queries // (num_tasks - task)) % N

        # Track the actual queries added for this task
        queries_added_for_task = 0

        for i, cls in enumerate(selected_classes):
            class_indices = (labels == cls).nonzero(as_tuple=True)[0]
            shuffled_indices = local_random.permutation(class_indices.numpy())

            # Ensure train set has exactly K samples
            train_indices = shuffled_indices[:K]

            # Query set allocation: Evenly distribute base_Q and assign 1 extra if remainder exists
            Q_cls = base_Q + (1 if i < remaining_per_task else 0)
            if include_val:
                val_indices = shuffled_indices[K : 2 * K]
                few_shot_val_masks[task][val_indices] = True
                test_indices = shuffled_indices[2 * K : 2 * K + Q_cls]
            else:
                test_indices = shuffled_indices[K : K + Q_cls]

            # Update masks for this task
            few_shot_train_masks[task][train_indices] = True
            few_shot_test_masks[task][test_indices] = True

            # Track actual queries added
            queries_added_for_task += len(test_indices)

        # Deduct the queries added from remaining queries
        remaining_queries -= queries_added_for_task

        selected_classes = np.unique(categories[few_shot_train_masks[task]])
        selected_classes_list.append(selected_classes)
        print(f"Task {task} selected classes: {selected_classes}")
        print(f"  Train samples: {few_shot_train_masks[task].count_nonzero()}")
        print(f"  Val samples: {few_shot_val_masks[task].count_nonzero()}")
        print(
            f"  Test samples: {few_shot_test_masks[task].count_nonzero()} (Queries added: {queries_added_for_task})"
        )
        print(f"  Remaining queries: {remaining_queries}")

    # Final check: Remaining queries should be 0
    # assert (
    #     remaining_queries == 0
    # ), f"Total query size was not correctly allocated! Remaining: {remaining_queries}"

    if remaining_queries > 0:
        print(f"\n WARNING: Remaining queries: {remaining_queries}")
        # * ensure remaining queries is at least N
        remaining_queries = max(remaining_queries, N * K)

    # If there are remaining queries, create additional tasks to use them up
    while remaining_queries > 0:
        task = len(selected_classes_list)  # New task index
        print(
            f"\nCreating additional Task {task} for remaining {remaining_queries} queries"
        )

        # Initialize masks for new task
        few_shot_train_masks = torch.cat(
            [few_shot_train_masks, torch.zeros((1, len(labels)), dtype=torch.bool)]
        )
        few_shot_val_masks = torch.cat(
            [few_shot_val_masks, torch.zeros((1, len(labels)), dtype=torch.bool)]
        )
        few_shot_test_masks = torch.cat(
            [few_shot_test_masks, torch.zeros((1, len(labels)), dtype=torch.bool)]
        )

        # Randomly select N classes for this task
        selected_classes = local_random.choice(valid_classes, size=N, replace=False)

        # Track queries added for this task
        queries_added_for_task = 0
        base_Q = remaining_queries // N
        remaining_per_task = remaining_queries % N

        for i, cls in enumerate(selected_classes):
            class_indices = (labels == cls).nonzero(as_tuple=True)[0]
            shuffled_indices = local_random.permutation(class_indices.numpy())

            # Ensure train set has exactly K samples
            train_indices = shuffled_indices[:K]

            # Calculate queries for this class (similar logic as before)
            Q_cls = min(
                base_Q + (1 if i < remaining_per_task else 0), remaining_queries
            )

            if include_val:
                val_indices = shuffled_indices[K : 2 * K]
                few_shot_val_masks[task][val_indices] = True
                test_indices = shuffled_indices[2 * K : 2 * K + Q_cls]
            else:
                test_indices = shuffled_indices[K : K + Q_cls]

            # Update masks
            few_shot_train_masks[task][train_indices] = True
            few_shot_test_masks[task][test_indices] = True

            queries_added_for_task += len(test_indices)
        remaining_queries -= queries_added_for_task

        selected_classes = np.unique(categories[few_shot_train_masks[task]])
        selected_classes_list.append(selected_classes)
        print(f"Task {task} selected classes: {selected_classes}")
        print(f"  Train samples: {few_shot_train_masks[task].count_nonzero()}")
        print(f"  Val samples: {few_shot_val_masks[task].count_nonzero()}")
        print(
            f"  Test samples: {few_shot_test_masks[task].count_nonzero()} (Queries added: {queries_added_for_task})"
        )
        print(f"  Remaining queries: {remaining_queries}")

    # Add multitask masks to graph
    graph.ndata["few_shot_train_mask"] = few_shot_train_masks.T
    graph.ndata["few_shot_val_mask"] = few_shot_val_masks.T
    graph.ndata["few_shot_test_mask"] = few_shot_test_masks.T

    return graph, selected_classes_list
