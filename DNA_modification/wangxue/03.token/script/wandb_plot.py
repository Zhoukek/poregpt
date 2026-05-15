#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wandb
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Upload images to Weights & Biases")
    parser.add_argument("--img_dir", required=True, help="Directory containing images")
    parser.add_argument("--project", default="signal_plot", help="W&B project name")
    return parser.parse_args()


def main():
    args = parse_args()

    wandb.init(project=args.project)

    if not os.path.isdir(args.img_dir):
        raise ValueError(f"Invalid directory: {args.img_dir}")

    for fname in os.listdir(args.img_dir):
        if fname.endswith(".png"):
            path = os.path.join(args.img_dir, fname)

            wandb.log({
                fname: wandb.Image(path)
            })

            print(f"[INFO] uploaded: {fname}")

    wandb.finish()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
