import datetime
import gc
import logging
import os
import pprint
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

import wandb
from src.lc_sampler import (
    load_single_dataset,
)
from model import build_model
from model.fewshot import fewshot_with_llm
from model.fewshot_classifier import FewShotClassifier
from model.quantizer import VQ
from src.config import load_config
from src.utils import (
    Tee,
    build_args,
    load_args_config,
    set_random_seed,
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


def main(args, ways=None, shots=None):
    if args.device < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
    shots = shots or args.k_shot
    shots = shots if isinstance(shots, list) else [shots]
    ways = ways or args.n_way
    ways = ways if isinstance(ways, list) else [ways]

    test_dataset_names = args.test_dataset  # * set of test dataset
    test_dataset_names = (
        test_dataset_names
        if isinstance(test_dataset_names, list)
        else [test_dataset_names]
    )
    test_ego_graph_file_paths = args.test_ego_graph_file_path
    test_ego_graph_file_paths = (
        test_ego_graph_file_paths
        if isinstance(test_ego_graph_file_paths, list)
        else [test_ego_graph_file_paths]
    )
    data_dir = args.data_dir

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

    # fix one seed for pretraining
    set_random_seed(0)

    args.num_features = 768
    args.out_dim = (
        args.num_features
        if cfg.get("out_dim", None) in [0, None]
        else cfg.get("out_dim")
    )
    if not printed:
        pp.pprint(vars(args))
        printed = True

    model = build_model(args, quantizer)
    load_model(model, args.checkpoint_path)
    print(f"Loading Model from {args.checkpoint_path}.")
    model = model.to(device)
    model.eval()

    # * LLM classifier
    fs_classifier = FewShotClassifier(args.llm_model, device)
    print("LLM used for classifier: {}".format(args.llm_model))

    # * few-shot learning
    val_accs = {}
    test_accs = {}

    for dataset_idx, test_dataset_name in enumerate(test_dataset_names):
        logging.info(f"---- start evaluation on {test_dataset_name} ----")
        val_accs[test_dataset_name] = {
            way: {shot: [] for shot in shots} for way in ways
        }
        test_accs[test_dataset_name] = {
            way: {shot: [] for shot in shots} for way in ways
        }

        # * test dataset
        (
            test_feats,
            test_graph,
            (test_num_features, test_num_classes),
            test_labels,
            test_categories,
            test_nodes,
            split_idx,
            split_labels,
            split_nodes,
        ) = load_single_dataset(
            test_dataset_name,
            data_dir,
            test_ego_graph_file_paths[dataset_idx],
            0,
        )
        torch.cuda.empty_cache()

        for way in ways:
            for shot in shots:
                print(f"####### Run {way}-way {shot}-shot")

                test_accs_list, val_accs_list = fewshot_with_llm(
                    model,
                    fs_classifier,
                    test_graph,
                    test_feats,
                    test_nodes,
                    test_categories,
                    way,
                    shot,
                    4,
                    device,
                )
                val_accs[test_dataset_name][way][shot] = val_accs_list
                test_accs[test_dataset_name][way][shot] = test_accs_list

                val_mean = np.mean(val_accs_list)
                val_std = np.std(val_accs_list)
                test_mean = np.mean(test_accs_list)
                test_std = np.std(test_accs_list)
                # Log metrics as regular scalars first
                metric_name = f"{test_dataset_name}/{way}-way_{shot}-shot"
                run.log(
                    {
                        f"{metric_name}/val_acc": val_mean,
                        f"{metric_name}/val_std": val_std,
                        f"{metric_name}/test_acc": test_mean,
                        f"{metric_name}/test_std": test_std,
                    }
                )

                # * Cleanup
                torch.cuda.empty_cache()
                gc.collect()

    # Print average validation and test accuracies across all datasets
    print("* Average Accuracies")
    for way in ways:
        for shot in shots:
            val_acc = np.mean(
                [
                    acc
                    for dataset_dict in val_accs.values()
                    for acc in dataset_dict[way][shot]
                ]
            )
            val_std = np.std(
                [
                    acc
                    for dataset_dict in val_accs.values()
                    for acc in dataset_dict[way][shot]
                ]
            )
            print(f"- {way}-Way {shot}-Shot Val Acc: {val_acc:.4f}±{val_std:.4f}")
            test_acc = np.mean(
                [
                    acc
                    for dataset_dict in test_accs.values()
                    for acc in dataset_dict[way][shot]
                ]
            )
            test_std = np.std(
                [
                    acc
                    for dataset_dict in test_accs.values()
                    for acc in dataset_dict[way][shot]
                ]
            )
            print(f"- {way}-Way {shot}-Shot Test Acc: {test_acc:.4f}±{test_std:.4f}")

    # Finalize the run
    run.finish()

    # Remove the wandb directory
    wandb_dir = Path(run.dir).parent
    if os.path.exists(wandb_dir):
        shutil.rmtree(wandb_dir)
        print(f"Removed wandb directory {wandb_dir}")

    # Return appropriate results
    return test_accs, val_accs


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

    test_accs, val_accs = main(args)
    ways = args.n_way if isinstance(args.n_way, list) else [args.n_way]
    shots = args.k_shot if isinstance(args.k_shot, list) else [args.k_shot]
    # * save the results in csv
    datasets = "-".join(args.dataset) if len(args.dataset) else args.dataset
    test_datasets = (
        args.test_dataset
        if isinstance(args.test_dataset, list)
        else [args.test_dataset]
    )
    csv_dir = Path(args.cfg_path).parent / "csv" / datasets
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = f"{csv_dir}/{cfg.config_name}_{start_time}.csv"
    with open(csv_file, "w") as f:
        f.write("Dataset,Way,Shot,Val Acc,Test Acc\n")
        for test_data in test_datasets:
            for way in ways:
                for shot in shots:
                    val_mean = np.mean(val_accs[test_data][way][shot])
                    val_std = np.std(val_accs[test_data][way][shot])
                    test_mean = np.mean(test_accs[test_data][way][shot])
                    test_std = np.std(test_accs[test_data][way][shot])
                    f.write(
                        f"{test_data},{way},{shot},{val_mean:.4f}±{val_std:.4f},{test_mean:.4f}±{test_std:.4f}\n"
                    )
