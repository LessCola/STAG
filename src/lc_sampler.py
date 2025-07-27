import logging
import os

import dgl
import numpy as np
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from torch.utils.data import DataLoader

from src.utils import mask_edge

from .gen_data import get_dgl_graph, preprocess, scale_feats

torch.multiprocessing.set_sharing_strategy("file_system")  # default

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class LinearProbingDataLoader(DataLoader):
    def __init__(self, idx, feats, labels=None, **kwargs):
        self.labels = labels
        self.feats = feats

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=idx, **kwargs)

    def __collate_fn__(self, batch_idx):
        feats = self.feats[batch_idx]
        label = self.labels[batch_idx]

        return feats, label


# * single-dataset loader
class SingleLCLoader(DataLoader):
    def __init__(
        self, root_nodes, graph, feats, labels=None, drop_edge_rate=0, **kwargs
    ):
        self.graph = graph
        self.labels = labels
        if self.labels is not None:
            self.labels = np.array(self.labels).flatten()
        self._drop_edge_rate = drop_edge_rate
        # ! ego_graph_nodes: subgraphs for each ego node
        self.ego_graph_nodes = root_nodes
        self.feats = feats

        dataset = np.arange(len(root_nodes))
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def drop_edge(self, g):
        if self._drop_edge_rate <= 0:
            return g, g

        g = g.remove_self_loop()
        mask_index1 = mask_edge(g, self._drop_edge_rate)
        mask_index2 = mask_edge(g, self._drop_edge_rate)
        g1 = dgl.remove_edges(g, mask_index1).add_self_loop()
        g2 = dgl.remove_edges(g, mask_index2).add_self_loop()
        return g1, g2

    def __collate_fn__(self, batch_idx):
        ego_nodes = [self.ego_graph_nodes[i] for i in batch_idx]
        subgs = [self.graph.subgraph(ego_nodes[i]) for i in range(len(ego_nodes))]
        modified_subgs = [g.remove_self_loop().add_self_loop() for g in subgs]
        # ! sg: batched subgraphs
        # sg = dgl.batch(subgs)
        # * to maintain the same batch size
        sg = dgl.batch(modified_subgs)
        # print("batch size of sg: ", sg.batch_size)

        # ! nodes: total nodes in each batch
        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long()
        # ! num_nodes: number of nodes in each ego graph
        num_nodes = [x.shape[0] for x in ego_nodes]
        # ! cum_num_nodes: index of the first node (ego node) in each ego graph
        cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]

        if self._drop_edge_rate > 0:
            drop_g1, drop_g2 = self.drop_edge(sg)

        # ? sg = sg.remove_self_loop().add_self_loop()
        # ? print("batch size of sg after processing: ", sg.batch_size) # here batch size is 1
        sg.ndata["feat"] = self.feats[nodes]
        # ! targets: the index of ego node in the batch
        targets = torch.from_numpy(cum_num_nodes)

        if self.labels is not None:
            labels = self.labels[batch_idx]
        else:
            labels = None

        if self._drop_edge_rate > 0:
            return sg, targets, labels, nodes, drop_g1, drop_g2
        else:
            return sg, targets, labels, nodes


# * multi-dataset loader
class MultiDatasetLCLoader(DataLoader):
    def __init__(
        self,
        nodes,  # Single list of subgraphs or list of lists
        all_graphs,
        all_feats,
        dataset_idx,  # Single list of dataset indices or list of lists
        labels=None,
        drop_edge_rate=0,
        **kwargs,
    ):
        self.all_graphs = all_graphs
        self.all_feats = all_feats
        self.labels = labels
        self._drop_edge_rate = drop_edge_rate

        # Check if dataset_idx is a list of lists (e.g., [dataset_idx_train, dataset_idx_valid])
        if isinstance(dataset_idx[0], list):
            # Flatten the list of lists into a single list
            self.nodes = [node for sublist in nodes for node in sublist]
            self.dataset_idx = [idx for sublist in dataset_idx for idx in sublist]
            if self.labels is not None:
                self.labels = np.concatenate(labels)
        else:
            # nodes is already a single list
            self.nodes = nodes
            self.dataset_idx = dataset_idx
            if self.labels is not None:
                self.labels = np.array(self.labels).flatten()

        # Debugging: Ensure node and dataset index lists have the same length
        assert len(self.nodes) == len(self.dataset_idx), (
            "Mismatch between the number of nodes and dataset indices."
        )

        dataset = np.arange(len(self.nodes))  # Treat each subgraph as a sample
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def drop_edge(self, g):
        if self._drop_edge_rate <= 0:
            return g, g

        g = g.remove_self_loop()
        mask_index1 = mask_edge(g, self._drop_edge_rate)
        mask_index2 = mask_edge(g, self._drop_edge_rate)
        g1 = dgl.remove_edges(g, mask_index1).add_self_loop()
        g2 = dgl.remove_edges(g, mask_index2).add_self_loop()
        return g1, g2

    def __collate_fn__(self, batch_idx):
        # Get the ego nodes for the current batch
        ego_nodes = [self.nodes[i] for i in batch_idx]
        batch_dataset_idx = [
            self.dataset_idx[i] for i in batch_idx
        ]  # Get dataset indices for the current batch

        # Debugging: Check if all ego_nodes indices are valid
        for i in range(len(ego_nodes)):
            graph_idx = batch_dataset_idx[i]
            graph = self.all_graphs[graph_idx]
            max_node = graph.num_nodes()
            invalid_nodes = [n for n in ego_nodes[i] if n < 0 or n >= max_node]
            if invalid_nodes:
                raise ValueError(
                    f"Invalid nodes found in batch {i} for graph {graph_idx}: {invalid_nodes}"
                )

        # Select the corresponding graph for each subgraph based on dataset index
        subgs = [
            self.all_graphs[batch_dataset_idx[i]].subgraph(ego_nodes[i])
            for i in range(len(ego_nodes))
        ]
        modified_subgs = [g.remove_self_loop().add_self_loop() for g in subgs]

        # Batch the subgraphs
        sg = dgl.batch(modified_subgs)

        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long()
        num_nodes = [x.shape[0] for x in ego_nodes]
        cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]

        if self._drop_edge_rate > 0:
            drop_g1, drop_g2 = self.drop_edge(sg)

        # Assign node features, selecting the correct features based on dataset index
        sg.ndata["feat"] = torch.cat(
            [
                self.all_feats[batch_dataset_idx[i]][ego_nodes[i]]
                for i in range(len(batch_idx))
            ],
            dim=0,
        )

        targets = torch.from_numpy(cum_num_nodes)

        if self.labels is not None:
            labels = self.labels[batch_idx]
        else:
            labels = None

        if self._drop_edge_rate > 0:
            return sg, targets, labels, nodes, drop_g1, drop_g2
        else:
            return sg, targets, labels, nodes


# * setup single-dataset loader
def setup_single_dataset_dataloader(
    loader_type,
    training_nodes,
    graph,
    feats,
    batch_size,
    drop_edge_rate=0,
    pretrain_clustergcn=False,
    cluster_iter_data=None,
):
    num_workers = 8

    if loader_type in ["lc", "k_hop"]:
        assert training_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    # print(" -------- drop edge rate: {} --------".format(drop_edge_rate))
    dataloader = SingleLCLoader(
        training_nodes,
        graph,
        feats=feats,
        drop_edge_rate=drop_edge_rate,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        # persistent_workers=True if num_workers > 0 else False,
        persistent_workers=False,
        num_workers=num_workers,
        worker_init_fn=lambda worker_id: np.random.seed(worker_id),
    )
    return dataloader


def _setup_eval_dataloader(
    loader_type, graph, feats, ego_graph_nodes=None, batch_size=128, shuffle=False
):
    num_workers = 8
    if loader_type in ["lc", "k_hop"]:
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    dataloader = SingleLCLoader(
        ego_graph_nodes,
        graph,
        feats,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        # persistent_workers=True if num_workers > 0 else False,
        persistent_workers=False,
        num_workers=num_workers,
        worker_init_fn=lambda worker_id: np.random.seed(worker_id),
    )
    return dataloader


# original version
def setup_eval_dataloader(
    loader_type,
    graph,
    feats,
    ego_graph_nodes=None,
    batch_size=128,
    shuffle=False,
    num_workers=8,
):
    if loader_type in ["lc", "k_hop"]:
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    dataloader = SingleLCLoader(
        ego_graph_nodes,
        graph,
        feats,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        # persistent_workers=True if num_workers > 0 else False,
        persistent_workers=False,
        num_workers=num_workers,
        pin_memory=True,  # Enable pinned memory for faster data transfer
        worker_init_fn=lambda worker_id: np.random.seed(worker_id),
    )
    return dataloader


# * eval dataloader with small size
def setup_eval_small_dataloader(
    loader_type,
    graph,
    feats,
    ego_graph_nodes=None,
    batch_size=32,  # Small batch size
    shuffle=False,
    num_workers=0,  # Single-threaded for small data
):
    def _custom_collate_fn(batch):
        """
        Custom collate function to handle preloaded data and batch it efficiently.

        Args:
            batch: A list of preloaded data items. Each item in the batch is a tuple:
                (subgraph, targets, labels, nodes).

        Returns:
            Batched subgraph, targets, labels (if present), and nodes.
        """
        # Collect individual components from the batch
        subgs = [item[0] for item in batch]  # Subgraphs
        targets = [item[1] for item in batch]  # Targets (ego node indices)
        labels = (
            [item[2] for item in batch] if batch[0][2] is not None else None
        )  # Labels
        nodes = [item[3] for item in batch]  # Node IDs in the subgraphs

        # Batch the subgraphs using DGL
        batched_subgraph = dgl.batch(subgs)

        # Concatenate the targets and node IDs
        batched_targets = torch.cat(targets, dim=0)
        batched_nodes = torch.cat(nodes, dim=0)

        # Batch the labels if they exist
        if labels is not None:
            batched_labels = torch.cat(
                [label for label in labels if label is not None], dim=0
            )
        else:
            batched_labels = None

        return batched_subgraph, batched_targets, batched_labels, batched_nodes

    if loader_type in ["lc", "k_hop"]:
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    # Preload data if small and feasible
    preloaded_data = list(
        SingleLCLoader(
            ego_graph_nodes,
            graph,
            feats,
            batch_size=len(ego_graph_nodes),
            shuffle=shuffle,
            drop_last=False,
        )
    )

    dataloader = torch.utils.data.DataLoader(
        preloaded_data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,  # Optimize data transfer to GPU
        num_workers=num_workers,  # Single-threaded
        collate_fn=_custom_collate_fn,
    )

    return dataloader


# * multi-dataset loader
def setup_multi_datasets_dataloader(
    loader_type,
    nodes,
    all_graphs,
    all_feats,
    batch_size,
    shuffle,
    dataset_idx,
    drop_edge_rate=0,
    pretrain_clustergcn=False,
    cluster_iter_data=None,
    labels=None,
):
    num_workers = 8

    if loader_type in ["lc", "k_hop"]:
        assert nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    # Multi-dataset loader
    dataloader = MultiDatasetLCLoader(
        nodes=nodes,
        all_graphs=all_graphs,
        all_feats=all_feats,
        dataset_idx=dataset_idx,
        drop_edge_rate=drop_edge_rate,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        persistent_workers=False,
        num_workers=num_workers,
        worker_init_fn=lambda worker_id: np.random.seed(worker_id),
        labels=labels,
    )

    return dataloader


# * use sentence transformer for node embeddings
def get_dataset(
    data_dir,
    dataset_name,
    seed=0,
):
    if dataset_name in [
        "cora",
        "citeseer",
        "pubmed",
        "wiki-cs",
        "ogbn-arxiv",
        "arxiv-2023",
        "ogbn-products",
        "cora_full",
    ]:
        print("load dataset: ", dataset_name)
        graph, (num_features, num_classes), categories = get_dgl_graph(
            dataset_name,
        )
        feats = graph.ndata["feat"]
        labels = graph.ndata["label"]
        # * real split indices
        train_mask, val_mask, test_mask = (
            graph.ndata["train_mask"],
            graph.ndata["val_mask"],
            graph.ndata["test_mask"],
        )
        train_indices = torch.nonzero(train_mask, as_tuple=False).squeeze()
        val_indices = torch.nonzero(val_mask, as_tuple=False).squeeze()
        test_indices = torch.nonzero(test_mask, as_tuple=False).squeeze()
        split_idx = {
            "train": train_indices,
            "valid": val_indices,
            "test": test_indices,
        }
        if dataset_name == "ogbn-arxiv":
            labels = labels.view(-1)
        return (
            feats,
            graph,
            (num_features, num_classes),
            labels,
            categories,
            split_idx,
        )

    elif dataset_name == "ogbn-papers100M":  # dataset_name.startswith("ogbn"):
        print("load ogbn dataset: ", dataset_name)
        dataset = DglNodePropPredDataset(
            dataset_name, root=os.path.join(data_dir, "dataset")
        )
        graph, labels = dataset[0]

        if "year" in graph.ndata:
            del graph.ndata["year"]
        if not graph.is_multigraph:
            logging.info("--- to undirected graph ---")
            graph = preprocess(graph)
        graph = graph.remove_self_loop().add_self_loop()

        split_idx = dataset.get_idx_split()
        labels = labels.view(-1)

        feats = graph.ndata.pop("feat")
        if dataset_name in ("ogbn-arxiv", "ogbn-papers100M"):
            feats = scale_feats(feats)
    elif dataset_name == "mag-scholar-f":
        edge_index = np.load(os.path.join(data_dir, dataset_name, "edge_index_f.npy"))
        feats = torch.from_numpy(
            np.load(os.path.join(data_dir, "feature_f.npy"))
        ).float()

        graph = dgl.DGLGraph((edge_index[0], edge_index[1]))

        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        labels = torch.from_numpy(np.load(os.path.join(data_dir, "label_f.npy"))).to(
            torch.long
        )
        split_idx = torch.load(os.path.join(data_dir, "split_idx_f.pt"))
    return feats, graph, labels, None, split_idx, (None, None), None


# * single-dataset
def load_single_dataset(
    dataset_name,
    data_dir,
    ego_graphs_file_path,
    seed=0,
):
    (
        feats,
        graph,
        (num_features, num_classes),
        labels,
        categories,
        split_idx,
    ) = get_dataset(data_dir, dataset_name, seed)

    train_lbls = labels[split_idx["train"]]
    val_lbls = labels[split_idx["valid"]]
    test_lbls = labels[split_idx["test"]]

    split_labels = torch.cat([train_lbls, val_lbls, test_lbls])

    os.makedirs(os.path.dirname(ego_graphs_file_path), exist_ok=True)

    if not os.path.exists(ego_graphs_file_path):
        raise FileNotFoundError(f"{ego_graphs_file_path} doesn't exist")
    else:
        nodes = torch.load(ego_graphs_file_path)

    if dataset_name in [
        "cora",
        "citeseer",
        "pubmed",
        "wiki-cs",
        "ogbn-arxiv",
        "arxiv-2023",
        "ogbn-products",
        "cora_full",
    ]:
        # * merge pseudo splits into one with the order of original nodes indices
        # nodes = nodes[0] + nodes[1] + nodes[2]
        nodes = sum([list(n) for n in nodes], [])
        # * update the subgraphs splits with the original data splits
        train_nodes = [nodes[i] for i in split_idx["train"]]
        val_nodes = [nodes[i] for i in split_idx["valid"]]
        test_nodes = [nodes[i] for i in split_idx["test"]]
        split_nodes = [train_nodes, val_nodes, test_nodes]

    elif not dataset_name.startswith("ogbn") and dataset_name != "mag-scholar-f":
        # * merge pseudo splits into one with the order of original nodes indices
        # nodes = nodes[0] + nodes[1] + nodes[2]
        nodes = sum([list(n) for n in nodes], [])
        train_nodes = [nodes[i] for i in split_idx["train"]]
        val_nodes = [nodes[i] for i in split_idx["valid"]]
        test_nodes = [nodes[i] for i in split_idx["test"]]
        split_nodes = [train_nodes, val_nodes, test_nodes]
    else:
        split_nodes = nodes
        nodes = sum([list(n) for n in nodes], [])

    # * nodes are the subgraphs for each ego node in order of original nodes indices
    return (
        feats,
        graph,
        (num_features, num_classes),
        labels,
        categories,
        nodes,
        split_idx,
        split_labels,
        split_nodes,
    )


# * combine multi-dataset
def load_multi_datasets(dataset_names, data_dir, ego_graphs_file_paths, seed=0):
    all_feats, all_graphs, all_labels = [], [], []
    all_num_features, all_num_classes = 0, 0
    combined_train_nodes, combined_val_nodes, combined_test_nodes = (
        [],
        [],
        [],
    )  # Separate lists for train, val, test
    dataset_idx_train, dataset_idx_val, dataset_idx_test = (
        [],
        [],
        [],
    )  # Dataset indices for train, val, test
    combined_split_idx = {"train": [], "valid": [], "test": []}  # Combined split_idx
    all_categories = []  # Combined categories
    node_offset = 0  # Offset to adjust node indices for each dataset
    label_offset = 0  # Offset to adjust labels for each dataset

    # * Dataset indices for all nodes in the combined dataset
    all_dataset_idx = []
    all_nodes = []

    # Initialize empty tensors for labels
    labels_train = torch.tensor([], dtype=torch.long)
    labels_valid = torch.tensor([], dtype=torch.long)
    labels_test = torch.tensor([], dtype=torch.long)

    for idx, (dataset_name, ego_graphs_file_path) in enumerate(
        zip(dataset_names, ego_graphs_file_paths)
    ):
        (
            feats,
            graph,
            (num_features, num_classes),
            labels,
            categories,
            nodes,
            split_idx,
            split_labels,
            split_nodes,
        ) = load_single_dataset(dataset_name, data_dir, ego_graphs_file_path, seed)
        all_feats.append(feats)
        all_graphs.append(graph)
        all_categories.extend(categories)
        all_num_features = num_features
        all_num_classes += num_classes

        # Adjust labels to account for label_offset
        # adjusted_labels = graph.ndata["label"] + label_offset
        adjusted_labels = labels + label_offset
        all_labels.append(adjusted_labels)
        # Concatenate the labels for train, valid, and test
        labels_train = torch.cat((labels_train, adjusted_labels[split_idx["train"]]))
        labels_valid = torch.cat((labels_valid, adjusted_labels[split_idx["valid"]]))
        labels_test = torch.cat((labels_test, adjusted_labels[split_idx["test"]]))

        # Separate subgraph lists for train, val, and test nodes
        train_nodes = split_nodes[0]
        val_nodes = split_nodes[1]
        test_nodes = split_nodes[2]

        # Add the train, val, and test subgraphs to their respective lists
        combined_train_nodes.extend(train_nodes)
        combined_val_nodes.extend(val_nodes)
        combined_test_nodes.extend(test_nodes)

        # Keep track of which dataset each subgraph belongs to (train, val, test)
        dataset_idx_train.extend([idx] * len(train_nodes))
        dataset_idx_val.extend([idx] * len(val_nodes))
        dataset_idx_test.extend([idx] * len(test_nodes))

        # * nodes are the subgraphs for each ego node in order of original nodes indices
        all_nodes.extend(nodes)
        all_dataset_idx.extend([idx] * len(nodes))

        # Adjust split indices with node_offset and record them
        combined_split_idx["train"].extend(
            [i + node_offset for i in split_idx["train"]]
        )
        combined_split_idx["valid"].extend(
            [i + node_offset for i in split_idx["valid"]]
        )
        combined_split_idx["test"].extend([i + node_offset for i in split_idx["test"]])

        # Update the node offset for future datasets
        node_offset += len(nodes)

        # Update label_offset for future datasets (max label from the current dataset)
        label_offset += adjusted_labels.max().item() + 1

    # Return feats, graphs, labels, and separate lists for train, val, test nodes, along with dataset indices
    return (
        all_feats,
        all_graphs,
        (all_num_features, all_num_classes),
        all_labels,
        all_categories,
        all_nodes,
        all_dataset_idx,
        combined_split_idx,  # Combined split_idx for all datasets
        [
            labels_train,
            labels_valid,
            labels_test,
        ],  # Return the concatenated labels for each split,
        [
            combined_train_nodes,
            combined_val_nodes,
            combined_test_nodes,
        ],  # Separated lists for nodes
        [
            dataset_idx_train,
            dataset_idx_val,
            dataset_idx_test,
        ],  # Dataset indices for each split
    )
