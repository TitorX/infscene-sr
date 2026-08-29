"""Train the patch-level SR3 conditional diffusion model."""
import argparse
from functools import partial
from pathlib import Path

# GDAL must be imported before torch. Torch loads the system libgcc, which is
# older than the one GDAL was built against, and GDAL then fails to load at all.
from osgeo import gdal  # noqa: F401  (import order matters)

import yaml
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import RSSRDataset
from sr3 import SR3LightningModule


torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SR3 for remote-sensing SR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-c", "--config", type=str, default="config/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Version directory or .ckpt file to resume from "
             "(e.g. runs/naip_5x/version_0)",
    )
    return parser.parse_args()


def setup_datasets(config):
    dataset_config = config["dataset"]

    create_dataset = partial(
        RSSRDataset,
        root_dir=dataset_config["root_dir"],
        patch_size=dataset_config["patch_size"],
        stride=dataset_config["stride"],
        lr_patch_size=dataset_config["lr_patch_size"],
        bands=dataset_config["bands"],
        overlap_threshold=dataset_config["overlap_threshold"],
        max_nodata_ratio=dataset_config["max_nodata_ratio"],
        nodata_value=dataset_config.get("nodata_value", None),
    )

    print("Creating training dataset...")
    train_dataset = create_dataset(mask_shapefile=dataset_config["masks"]["train"])

    print("Creating validation dataset...")
    val_dataset = create_dataset(mask_shapefile=dataset_config["masks"]["val"])

    test_dataset = None
    if dataset_config["masks"].get("test") is not None:
        print("Creating test dataset...")
        test_dataset = create_dataset(mask_shapefile=dataset_config["masks"]["test"])

    print(f"Patches: {len(train_dataset)} train, {len(val_dataset)} val", end="")
    print(f", {len(test_dataset)} test" if test_dataset else "")

    return train_dataset, val_dataset, test_dataset


def setup_dataloaders(train_dataset, val_dataset, test_dataset, config):
    dataloader_config = config["dataloader"]

    create_dataloader = partial(
        DataLoader,
        batch_size=dataloader_config["batch_size"],
        num_workers=dataloader_config["num_workers"],
        pin_memory=True,
        persistent_workers=dataloader_config["num_workers"] > 0,
        shuffle=False,
    )

    train_loader = create_dataloader(train_dataset, shuffle=True)
    val_loader = create_dataloader(val_dataset)
    test_loader = create_dataloader(test_dataset) if test_dataset else None

    return train_loader, val_loader, test_loader


def find_latest_checkpoint(resume_path):
    """Resolve a version directory to its most recently written checkpoint."""
    resume_dir = Path(resume_path)

    if not resume_dir.exists():
        raise ValueError(f"Resume path does not exist: {resume_path}")

    if resume_dir.is_file() and resume_dir.suffix == ".ckpt":
        return str(resume_dir)

    checkpoint_dir = resume_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise ValueError(f"No .ckpt files found in {checkpoint_dir}")

    return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_dataset, val_dataset, test_dataset = setup_datasets(config)
    train_loader, val_loader, test_loader = setup_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )

    model = SR3LightningModule(
        in_channels=len(config["dataset"]["bands"]),
        image_size=config["dataset"]["patch_size"],
    )

    logger = TensorBoardLogger(
        save_dir=config["experiment"].get("log_dir", "runs"),
        name=config["experiment"]["name"],
        default_hp_metric=False,
    )

    callbacks = [
        # No monitored metric: every epoch is kept, the last one is "best".
        ModelCheckpoint(
            save_top_k=-1,
            every_n_epochs=1,
            filename="epoch={epoch:02d}-step={step}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = find_latest_checkpoint(args.resume)
        print(f"Resuming from {resume_checkpoint}")

    trainer = L.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator="auto",
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        # Validation samples a full reverse chain, so one batch per epoch.
        limit_val_batches=1,
        precision="bf16-mixed",
        accumulate_grad_batches=config["trainer"].get("accumulate_grad_batches", 8),
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_checkpoint,
    )

    if test_loader is not None:
        trainer.test(model, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
