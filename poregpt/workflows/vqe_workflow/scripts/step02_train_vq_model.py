# test/test_vq_tokenizer/test_train_vq_model.py
from nanopore_signal_tokenizer import vq_train
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_dir", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, default="demo_nanopore_vq_tokenizer.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--codebook_size", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=12000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--loss_csv_path", type=str, default="train_loss.csv")
    parser.add_argument("--save_checkpoint_interval", type=int, default=10)
    parser.add_argument("--do_evaluate", action="store_true", help="Enable codebook usage evaluation")

    args = parser.parse_args()

    vq_train(
        npy_dir=args.npy_dir,
        output_model_path=args.output_model_path,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        codebook_size=args.codebook_size,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        do_evaluate=args.do_evaluate,
        commitment_weight=args.commitment_weight,
        loss_csv_path=args.loss_csv_path,
        save_checkpoint_interval=args.save_checkpoint_interval
    )

