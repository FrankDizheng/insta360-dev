"""
SmolVLA fine-tuning on SO-100 pick-and-place dataset.
Uses the pretrained lerobot/smolvla_base model and fine-tunes on
lerobot/svla_so100_pickplace (50 episodes, ~20k frames).

Usage:
    python train_smolvla_so100.py
    # or multi-GPU:
    accelerate launch --num_processes=4 train_smolvla_so100.py
"""

import logging
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed

import sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def log(msg):
    print(f"[TRAIN] {msg}", flush=True)

OUTPUT_DIR = Path("/data/xinna/robot-arm/outputs/smolvla_so100_pickplace")
PRETRAINED_PATH = "lerobot/smolvla_base"
DATASET_REPO = "lerobot/svla_so100_pickplace"

TRAINING_STEPS = 10000
BATCH_SIZE = 8
LOG_FREQ = 50
SAVE_FREQ = 2000
NUM_WORKERS = 4


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    is_main = accelerator.is_main_process
    device = accelerator.device

    log(f"Device: {device}, Num processes: {accelerator.num_processes}, is_main: {is_main}")

    set_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- Build config ---
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DATASET_REPO),
        output_dir=OUTPUT_DIR,
        steps=TRAINING_STEPS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        log_freq=LOG_FREQ,
        save_freq=SAVE_FREQ,
        eval_freq=-1,
        rename_map={
            "observation.images.top": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        },
    )
    cli_overrides = []
    cfg.policy = PreTrainedConfig.from_pretrained(PRETRAINED_PATH, cli_overrides=cli_overrides)
    cfg.policy.pretrained_path = Path(PRETRAINED_PATH)
    cfg.policy.push_to_hub = False
    cfg.validate()

    log(f"Output: {cfg.output_dir}")
    log(f"Dataset: {DATASET_REPO}, Pretrained: {PRETRAINED_PATH}")
    log(f"Steps: {TRAINING_STEPS}, Batch: {BATCH_SIZE}")

    # --- Dataset ---
    log("Loading dataset...")
    dataset = make_dataset(cfg)
    log(f"Dataset loaded: {len(dataset)} samples")

    # --- Policy ---
    log("Loading pretrained policy...")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    log(f"Model params: {total:,} total, {trainable:,} trainable ({100*trainable/total:.1f}%)")

    # --- Preprocessor ---
    log("Creating preprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        cfg.policy, dataset_stats=dataset.meta.stats, rename_map=cfg.rename_map
    )
    log("Preprocessor created")

    # --- Optimizer & Scheduler ---
    log("Creating optimizer...")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    log("Optimizer created")

    # --- DataLoader ---
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # --- Accelerate prepare ---
    log("Accelerate prepare...")
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    policy.train()
    log("Model in training mode")

    dl_iter = cycle(dataloader)

    # Rename map: dataset keys -> policy keys (applied to each batch)
    batch_rename = {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }

    log("=" * 60)
    log("Starting training!")
    log("=" * 60)

    start_time = time.time()
    for step in range(1, TRAINING_STEPS + 1):
        if step <= 3:
            log(f"Step {step}: fetching batch...")
        batch = next(dl_iter)
        if step <= 3:
            log(f"Step {step}: batch keys = {list(batch.keys())}")
        batch = preprocessor(batch)

        # Rename dataset image keys to match policy's expected input features
        for old_key, new_key in batch_rename.items():
            if old_key in batch:
                batch[new_key] = batch.pop(old_key)
            pad_old = f"{old_key}_is_pad"
            pad_new = f"{new_key}_is_pad"
            if pad_old in batch:
                batch[pad_new] = batch.pop(pad_old)

        loss_dict = policy.forward(batch)
        if isinstance(loss_dict, tuple):
            loss = loss_dict[0]
        else:
            loss = loss_dict["loss"]

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), cfg.policy.optimizer_grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        if step % LOG_FREQ == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_min = (TRAINING_STEPS - step) / steps_per_sec / 60
            lr = optimizer.param_groups[0]["lr"]
            log(
                f"step {step}/{TRAINING_STEPS} | "
                f"loss={loss.item():.4f} | "
                f"grad_norm={grad_norm.item():.4f} | "
                f"lr={lr:.2e} | "
                f"speed={steps_per_sec:.1f} step/s | "
                f"ETA={eta_min:.1f}min"
            )

        if step % SAVE_FREQ == 0:
            ckpt_dir = cfg.output_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(policy)
            unwrapped.save_pretrained(ckpt_dir)
            preprocessor.save_pretrained(ckpt_dir)
            postprocessor.save_pretrained(ckpt_dir)
            log(f"Saved checkpoint: {ckpt_dir}")

    # --- Final save ---
    final_dir = cfg.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(policy)
    unwrapped.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    elapsed = time.time() - start_time
    log("=" * 60)
    log(f"Training complete! {TRAINING_STEPS} steps in {elapsed/60:.1f} minutes")
    log(f"Final model saved to: {final_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
