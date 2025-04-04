"""
Train a diffusion model on images.
"""
import sys
import os
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from torch.utils.data.sampler import SubsetRandomSampler
from guided_diffusion.lidcloader import *
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
import torch
from guided_diffusion.train_util import TrainLoop

def main():
    args = create_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    use_distributed = args.ngpu > 1  # Check if we need distributed training

    if use_distributed:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=args.ngpu,
            rank=args.local_rank,
        )
        torch.cuda.set_device(args.local_rank)
        print(f"Running on GPU {args.local_rank} with DistributedDataParallel.")
    else:
        print("Running in single-GPU mode.")
    
    logger.configure()

    logger.log("Creating model, diffusion, prior and posterior distribution...")
    model, diffusion, prior, posterior = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    prior.to(device)
    if use_distributed:
        prior = torch.nn.SyncBatchNorm.convert_sync_batchnorm(prior)
        prior = torch.nn.parallel.DistributedDataParallel(
            prior, device_ids=[args.local_rank], output_device=args.local_rank
        )

    posterior.to(device)
    if use_distributed:
        posterior = torch.nn.SyncBatchNorm.convert_sync_batchnorm(posterior)
        posterior = torch.nn.parallel.DistributedDataParallel(
            posterior, device_ids=[args.local_rank], output_device=args.local_rank
        )

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    logger.log("Creating data loader...")
    ds = get_lidc_dataset(args.data_dir, train=True)

    if use_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=args.ngpu, rank=args.local_rank
        )
        dataloader = th.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, sampler=sampler
        )
    else:
        dataloader = th.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True
        )

    data = iter(dataloader)

    logger.log("Training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=dataloader,
        prior=prior,
        posterior=posterior,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="huynhspm/data/lidc",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ngpu', type=int, default=1)
   
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
