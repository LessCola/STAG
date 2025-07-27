import datetime
import gc
import logging
import os
import pprint
import random
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.lc_sampler import (
    load_multi_datasets,
    load_single_dataset,
    setup_multi_datasets_dataloader,
)
from model import build_model
from model.fewshot import fewshot_with_llm
from model.fewshot_classifier import FewShotClassifier
from model.quantizer import VQ
from src.config import load_config
from src.utils import (
    Tee,
    build_args,
    create_optimizer,
    load_args_config,
    set_random_seed,
    show_occupied_memory,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def save_model(model, filepath):
    model_states = {
        "encoder": model.encoder.state_dict(),
        "fusion": model.fusion.state_dict(),
        "enc_mask_token": model.enc_mask_token.data,  # Saving the data of the parameter directly
    }
    # Save the combined state dictionaries to the specified file
    torch.save(model_states, filepath)


def load_model(model, filepath):
    # Load the model states from the file
    model_states = torch.load(filepath)

    # Apply the state dictionaries to the respective components
    model.encoder.load_state_dict(model_states["encoder"])
    model.fusion.load_state_dict(model_states["fusion"])
    # Reload the enc_mask_token parameter
    if "enc_mask_token" in model_states:
        model.enc_mask_token.data = model_states["enc_mask_token"]


def pretrain(
    model,
    all_feats,
    all_graphs,
    pretrain_ego_graph_nodes,
    pretrain_dataset_idx,
    max_epoch,
    device,
    use_scheduler,
    lr,
    weight_decay,
    batch_size=512,
    sampling_method="lc",
    optimizer="adam",
    drop_edge_rate=0,
    model_dir="checkpoint",
    model_name="checkpoint.pt",
):
    global run, args

    model = model.to(device)
    optimizer = create_optimizer(optimizer, model, lr, weight_decay)

    # * training dataloader
    # * ego_graph_nodes: subgraphs for each ego node
    dataloader = setup_multi_datasets_dataloader(
        sampling_method,
        pretrain_ego_graph_nodes,
        all_graphs,
        all_feats,
        batch_size=batch_size,
        shuffle=True,
        dataset_idx=pretrain_dataset_idx,
        drop_edge_rate=drop_edge_rate,
    )

    logging.info(f"After creating dataloader: Memory: {show_occupied_memory():.2f} MB")
    if use_scheduler and max_epoch > 0:
        logging.info("Use scheduler")
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    # check if the sys.stdout is a terminal
    terminal_stdout = getattr(sys.stdout, "terminal_stdout", sys.stdout)
    for epoch in range(max_epoch):
        epoch_iter = tqdm(
            dataloader, file=terminal_stdout
        )  # sys.stdout.terminal_stdout
        losses = []

        for batch_g in epoch_iter:
            model.train()
            if drop_edge_rate > 0:
                # * sg, targets, label, nodes, drop_g1, drop_g2
                # * batch_g: batched subgraphs
                # * targets: the ego node of each ego graph
                # * nodes: all the nodes in each batch
                batch_g, targets, _, node_idx, drop_g1, drop_g2 = batch_g
                batch_g = batch_g.to(device)
                drop_g1 = drop_g1.to(device)
                drop_g2 = drop_g2.to(device)
                x = batch_g.ndata.pop("feat")
                loss_dict = model(batch_g, x, targets, epoch, drop_g1, drop_g2)
            else:
                batch_g, targets, _, node_idx = batch_g
                batch_g = batch_g.to(device)
                x = batch_g.ndata.pop("feat")
                loss_dict = model(batch_g, x, targets, epoch)
            loss = loss_dict["total_loss"]
            run.log(
                {
                    "train/{}".format(loss_): v.item()
                    if isinstance(v, torch.Tensor)
                    else v
                    for loss_, v in loss_dict.items()
                }
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            epoch_iter.set_description(
                f"Epoch {epoch + 1}/{max_epoch}, train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB"
            )
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        if args.save_model:
            save_model(model, os.path.join(model_dir, model_name))

        print(
            f"# Epoch {epoch + 1}/{max_epoch} | train_loss: {np.mean(losses):.4f}, Memory: {show_occupied_memory():.2f} MB"
        )
        run.log({"train/epoch_loss": np.mean(losses)})

        # * Cleanup after each epoch
        del batch_g, targets, node_idx, loss, x

    # * cleanup
    dataloader = None

    return model


def determine_sample_size(total_nodes):
    """
    Return the total number of nodes if it is less than or equal to 20,000.
    For larger datasets, cap the sample size at 20,000.
    """
    if total_nodes <= 30000:
        return total_nodes
    return min(total_nodes, 25000)


def update_training_data_with_balanced_sampling(all_nodes, all_dataset_idx):
    """
    Performs balanced sampling across multiple datasets while respecting size constraints.

    Parameters:
        all_nodes: List of subgraph nodes across all graphs
        all_dataset_idx: List of dataset indices for nodes

    Returns:
        Updated nodes and indices after balanced sampling
    """
    num_nodes = len(all_nodes)
    num_sampled = determine_sample_size(num_nodes)

    # If we need all samples or have single dataset, use simple sampling
    if num_sampled == num_nodes and len(set(all_dataset_idx)) == 1:
        return all_nodes, all_dataset_idx

    # Get counts and nodes per dataset
    unique_datasets = list(set(all_dataset_idx))
    dataset_nodes = {d: [] for d in unique_datasets}
    for node, idx in zip(all_nodes, all_dataset_idx):
        dataset_nodes[idx].append(node)

    # Calculate samples per dataset
    samples_per_dataset = num_sampled // len(unique_datasets)
    extra_samples = num_sampled % len(unique_datasets)

    # Sample from each dataset
    sampled_nodes = []
    sampled_indices = []

    for dataset_idx in unique_datasets:
        # Get sample size for this dataset
        sample_size = samples_per_dataset
        if extra_samples > 0:
            sample_size += 1
            extra_samples -= 1

        # Cap at available nodes
        available = len(dataset_nodes[dataset_idx])
        sample_size = min(sample_size, available)

        # Random sampling
        if sample_size > 0:
            sampled = random.sample(range(available), sample_size)
            sampled_nodes.extend([dataset_nodes[dataset_idx][i] for i in sampled])
            sampled_indices.extend([dataset_idx] * sample_size)

    return sampled_nodes, sampled_indices


def main(args, ways=None, shots=None):
    if args.device < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    shots = shots or args.k_shot
    shots = shots if isinstance(shots, list) else [shots]
    ways = ways or args.n_way
    ways = ways if isinstance(ways, list) else [ways]

    dataset_names = args.dataset  # * set of datasets
    test_dataset_names = args.test_dataset  # * set of test dataset
    test_dataset_names = (
        test_dataset_names
        if isinstance(test_dataset_names, list)
        else [test_dataset_names]
    )
    max_epoch = args.max_epoch
    drop_edge_rate = args.drop_edge_rate

    lr = args.lr
    weight_decay = args.weight_decay
    use_scheduler = args.scheduler
    batch_size = args.batch_size
    sampling_method = args.sampling_method
    ego_graph_file_paths = args.ego_graph_file_path
    test_ego_graph_file_paths = args.test_ego_graph_file_path
    test_ego_graph_file_paths = (
        test_ego_graph_file_paths
        if isinstance(test_ego_graph_file_paths, list)
        else [test_ego_graph_file_paths]
    )
    data_dir = args.data_dir
    optimizer_type = args.optimizer

    model_dir = "checkpoint"
    os.makedirs(model_dir, exist_ok=True)

    pp = pprint.PrettyPrinter()
    printed = False

    # * initialize quantizer
    quantizer = VQ(
        args,
        args.top_k,
        raw_temp=args.raw_temp,
        quant_temp=args.quant_temp,
        commit_score=args.commit_score,
    )

    # * ego_graph_nodes: subgraphs for each ego node
    (
        all_feats,
        all_graphs,
        (all_num_features, all_num_classes),
        all_labels,
        all_categories,
        all_nodes,
        all_dataset_idx,
        combined_split_idx,
        combined_labels,
        combined_ego_graph_nodes,
        combined_dataset_idx,
    ) = load_multi_datasets(dataset_names, data_dir, ego_graph_file_paths, 0)
    pretrain_dataset_idx = all_dataset_idx
    pretrain_ego_graph_nodes = all_nodes

    # fix one seed for pretraining
    set_random_seed(0)
    pretrain_ego_graph_nodes, pretrain_dataset_idx = (
        update_training_data_with_balanced_sampling(
            all_nodes,
            all_dataset_idx,
        )
    )

    if isinstance(all_feats, list):
        args.num_features = all_feats[0].shape[1]
    else:
        args.num_features = all_feats.shape[1]
    # * out_dim
    args.out_dim = (
        args.num_features
        if cfg.get("out_dim", None) in [0, None]
        else cfg.get("out_dim")
    )
    if not printed:
        pp.pprint(vars(args))
        printed = True

    if isinstance(cfg.ego_graph_file_path, list):
        ego_size = re.findall(r"\d+", cfg.ego_graph_file_path[0])[-1]
    else:
        ego_size = re.findall(r"\d+", cfg.ego_graph_file_path)[-1]

    model_name = f"{args.config_name}_{ego_size}_checkpoint.pt"
    model = build_model(args, quantizer)

    # ------------- pretraining starts ----------------
    logging.info("---- start pretraining ----")
    model = pretrain(
        model,
        all_feats,
        all_graphs,
        pretrain_ego_graph_nodes,
        pretrain_dataset_idx,
        max_epoch=max_epoch,
        device=device,
        use_scheduler=use_scheduler,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        drop_edge_rate=drop_edge_rate,
        sampling_method=sampling_method,
        optimizer=optimizer_type,
        model_dir=model_dir,
        model_name=model_name,
    )

    model = model.cpu()
    if args.save_model:
        logging.info(f"saving model to {model_dir}/{model_name}...")
        save_model(model, os.path.join(model_dir, model_name))
    # ------------- pretraining ends ----------------

    # Finalize the run
    run.finish()

    # Remove the wandb directory
    wandb_dir = Path(run.dir).parent
    if os.path.exists(wandb_dir):
        shutil.rmtree(wandb_dir)
        print(f"Removed wandb directory {wandb_dir}")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_args_config(args, args.cfg_path)
        cfg = load_config(args.cfg_path)

    run = wandb.init(project=cfg.project, config=cfg, name=cfg.config_name)
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Redirect stdout and stderr
    # create log dir under the parent directory of the config file
    os.makedirs(f"{Path(args.cfg_path).parent}/log", exist_ok=True)
    # Truncate config name if needed to ensure start_time fits
    max_config_len = 255 - len("_.log") - len(start_time)
    config_name = (
        cfg.config_name[:max_config_len]
        if len(cfg.config_name) > max_config_len
        else cfg.config_name
    )
    log_file = f"{Path(args.cfg_path).parent}/log/{config_name}_{start_time}.log"
    sys.stdout = Tee(log_file)  # open(log_file, "w")
    sys.stderr = sys.stdout

    main(args)
