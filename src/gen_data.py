import os
import pickle

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric as pyg
from sklearn.preprocessing import StandardScaler


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

    # return graph, selected_classes_list
    return few_shot_train_masks, few_shot_val_masks, few_shot_test_masks


def dgl_to_pyg(
    dgl_graph, raw_texts, category_names, n_way, k_shot, num_tasks=20, q_query=2000
):
    """Convert a DGL graph to a PyG graph."""
    # Move graph and all its attributes to CPU
    dgl_graph = dgl_graph.to("cpu")
    for key in dgl_graph.ndata:
        dgl_graph.ndata[key] = dgl_graph.ndata[key].to("cpu")
    for key in dgl_graph.edata:
        dgl_graph.edata[key] = dgl_graph.edata[key].to("cpu")

    # Get node features
    x = dgl_graph.ndata["feat"]
    # Get node labels
    y = dgl_graph.ndata["label"]
    # Remap y to be 0-indexed
    unique_labels = sorted(torch.unique(y).tolist())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = torch.tensor([label_map[label] for label in y.tolist()]).to("cpu")

    # * Create ordered label names based on the remapped y values
    sorted_labels = sorted(torch.unique(y).tolist())
    ordered_label_names = [None] * len(sorted_labels)  # Initialize with None

    for i, label in enumerate(sorted_labels):
        # Find the first occurrence of the label in y to get the corresponding category name
        ordered_label_names[i] = category_names[
            (y == label).nonzero(as_tuple=True)[0][0].item()
        ]

    # Get edge indices
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)

    # * create few shot masks
    train_masks, val_masks, test_masks = (
        create_few_shot_multitask_masks_fixed_query_size(
            dgl_graph,
            category_names,
            n_way,
            k_shot,
            q_query,
            num_tasks,
            include_val=True,
        )
    )

    # Create initial PyG Data dictionary
    data_dict = {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "raw_text": raw_texts,
        "label_names": ordered_label_names,
        "category_names": category_names,
    }

    data_dict.update(
        {"train_masks": train_masks, "val_masks": val_masks, "test_masks": test_masks}
    )

    return pyg.data.Data(**data_dict)


def get_cora_graph(root):
    root_path = os.path.join(root, "cora")
    # Load your DGL graph and metadata
    # * roberta-base-nli-stsb-mean-tokens
    dgl_graph = torch.load(os.path.join(root_path, "cora_graph.pth"))
    metadata = torch.load(os.path.join(root_path, "cora_metadata.pth"))
    with open(os.path.join(root_path, "cora_text.pkl"), "rb") as f:
        raw_texts = pickle.load(f)
    categories = metadata["categories"]
    # Convert to PyG
    # graph = dgl_to_pyg(dgl_graph, raw_texts, categories)
    # return graph
    return dgl_graph, raw_texts, categories


def get_cora_full_graph(root):
    root_path = os.path.join(root, "cora_full")
    # Load your DGL graph and metadata
    # * roberta-base-nli-stsb-mean-tokens
    dgl_graph = torch.load(os.path.join(root_path, "cora_full_graph.pth"))
    metadata = torch.load(os.path.join(root_path, "cora_full_metadata.pth"))
    raw_texts = []
    with open(os.path.join(root_path, "cora_full_text.txt"), "rb") as f:
        lines = f.readlines()
        for line in lines:
            line = line.decode("utf-8").strip().split("\t")
            raw_texts.append(line[2])

    categories = metadata["categories"]
    # Convert to PyG
    # graph = dgl_to_pyg(dgl_graph, raw_texts, categories)
    # return graph
    return dgl_graph, raw_texts, categories


def get_citeseer_graph(root):
    root_path = os.path.join(root, "citeseer")
    # Load your DGL graph and metadata
    # * roberta-base-nli-stsb-mean-tokens
    dgl_graph = torch.load(os.path.join(root_path, "citeseer_graph.pth"))
    metadata = torch.load(os.path.join(root_path, "citeseer_metadata.pth"))
    # with open(os.path.join(root_path, "citeseer_text.pkl"), "rb") as f:
    #     raw_texts = pickle.load(f)
    citeseer_metadata = torch.load(os.path.join(root_path, "citeseer_metadata.pt"))
    index_to_id = citeseer_metadata["index_to_id"]
    raw_texts = [
        citeseer_metadata["paper_content"][index_to_id[i]]
        for i in range(dgl_graph.num_nodes())
    ]
    categories = metadata["categories"]
    # Convert to PyG
    # graph = dgl_to_pyg(dgl_graph, raw_texts, categories)
    # return graph
    return dgl_graph, raw_texts, categories


def get_pubmed_graph(root):
    root_path = os.path.join(root, "pubmed")
    # Load your DGL graph and metadata
    # * roberta-base-nli-stsb-mean-tokens
    dgl_graph = torch.load(os.path.join(root_path, "pubmed_graph.pth"))
    metadata = torch.load(os.path.join(root_path, "pubmed_metadata.pth"))
    with open(os.path.join(root_path, "pubmed_text.pkl"), "rb") as f:
        raw_texts = pickle.load(f)
    categories = metadata["categories"]
    # Convert to PyG
    # graph = dgl_to_pyg(dgl_graph, raw_texts, categories)
    # return graph
    return dgl_graph, raw_texts, categories


def get_wikics_graph(root):
    root_path = os.path.join(root, "wiki-cs")
    # Load your DGL graph and metadata
    # * roberta-base-nli-stsb-mean-tokens
    dgl_graph = torch.load(os.path.join(root_path, "wiki-cs_graph.pth"))
    metadata = torch.load(os.path.join(root_path, "wiki-cs_metadata.pth"))
    with open(os.path.join(root_path, "wiki-cs_text.pkl"), "rb") as f:
        raw_texts = pickle.load(f)
    categories = metadata["categories"]
    # Convert to PyG
    # graph = dgl_to_pyg(dgl_graph, raw_texts, categories)
    # return graph
    return dgl_graph, raw_texts, categories


def get_arxiv_graph(root):
    root_path = os.path.join(root, "ogbn-arxiv")
    # Load your DGL graph and metadata
    # * roberta-base-nli-stsb-mean-tokens
    dgl_graph = torch.load(os.path.join(root_path, "ogbn-arxiv_graph.pth"))
    metadata = torch.load(os.path.join(root_path, "ogbn-arxiv_metadata.pth"))
    with open(os.path.join(root_path, "ogbn-arxiv_text.pkl"), "rb") as f:
        raw_texts = pickle.load(f)
    categories = metadata["categories"]
    # Convert to PyG
    # graph = dgl_to_pyg(dgl_graph, raw_texts, categories)
    # return graph
    return dgl_graph, raw_texts, categories


def get_products_graph(root):
    root_path = os.path.join(root, "ogbn-products")
    # Load your DGL graph and metadata
    # * roberta-base-nli-stsb-mean-tokens
    dgl_graph = torch.load(os.path.join(root_path, "ogbn-products_graph.pth"))
    metadata = torch.load(os.path.join(root_path, "ogbn-products_metadata.pth"))
    with open(os.path.join(root_path, "ogbn-products_text.pkl"), "rb") as f:
        raw_texts = pickle.load(f)
    categories = metadata["categories"]
    # Convert to PyG
    # graph = dgl_to_pyg(dgl_graph, raw_texts, categories)
    # return graph
    return dgl_graph, raw_texts, categories


def get_dgl_graph(dataset_name, root="./dataset"):
    if dataset_name == "cora":
        get_graph = get_cora_graph
    elif dataset_name == "cora_full":
        get_graph = get_cora_full_graph
    elif dataset_name == "citeseer":
        get_graph = get_citeseer_graph
    elif dataset_name == "pubmed":
        get_graph = get_pubmed_graph
    elif dataset_name == "wiki-cs":
        get_graph = get_wikics_graph
    elif dataset_name == "ogbn-arxiv":
        get_graph = get_arxiv_graph
    elif dataset_name == "ogbn-products":
        get_graph = get_products_graph
    dgl_graph, raw_texts, categories = get_graph(root)
    num_features = dgl_graph.ndata["feat"].shape[1]
    num_classes = len(np.unique(dgl_graph.ndata["label"]))
    return (
        dgl_graph,
        (num_features, num_classes),
        categories,
    )


def get_pyg_graph(dataset_name, n_way, k_shot, q_query, num_tasks, root="./dataset"):
    if dataset_name == "cora":
        get_graph = get_cora_graph
    elif dataset_name == "cora_full":
        get_graph = get_cora_full_graph
    elif dataset_name == "citeseer":
        get_graph = get_citeseer_graph
    elif dataset_name == "pubmed":
        get_graph = get_pubmed_graph
    elif dataset_name == "wiki-cs":
        get_graph = get_wikics_graph
    elif dataset_name == "arxiv":
        get_graph = get_arxiv_graph
    elif dataset_name == "products":
        get_graph = get_products_graph
    dgl_graph, raw_texts, categories = get_graph(root)
    graph = dgl_to_pyg(
        dgl_graph, raw_texts, categories, n_way, k_shot, num_tasks, q_query
    )
    num_nodes = dgl_graph.num_nodes()

    edge_index = graph.edge_index.to("cpu")
    # Convert edge_index to sparse adjacency matrix
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())),
        shape=(num_nodes, num_nodes),
    )
    # Convert to CSR format for efficient operations
    adj = adj.tocsr()

    features = graph.x.to("cpu").numpy()
    # Convert to scipy sparse matrix in CSR format
    features = sp.csr_matrix(features)

    labels = graph.y.to("cpu").numpy()
    # Convert labels to one-hot format
    num_classes = len(np.unique(labels))
    labels_onehot = np.zeros((labels.shape[0], num_classes))
    labels_onehot[np.arange(labels.shape[0]), labels] = 1
    labels = labels_onehot
    return adj, features, labels, graph.train_masks, graph.val_masks, graph.test_masks


def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    return graph


def scale_feats(x, scaler=None):
    if isinstance(x, torch.Tensor):
        feats = x.cpu().numpy()
    else:
        feats = x.numpy()
    if scaler is None:
        scaler = StandardScaler()
    if not hasattr(scaler, "scale_"):  # scaler is not fitted
        scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    # * save the scaler for later use on codebook
    return feats, scaler
