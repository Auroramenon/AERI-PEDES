import torch

from .lr_scheduler import LRSchedulerWithWarmup


def is_mask_generator_param(key):
    """Parameters of the dpm mask generator (CLS mask head or HMG)."""
    return "avm_mask_head" in key


def build_optimizer(args, model):
    params = []
    two_step = getattr(args, "avm_two_step", False)

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if two_step and is_mask_generator_param(key):
            # DPM two-step update: the mask generator has its own optimizer.
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        if "query" in key:
            # lr =  args.lr * args.lr_factor
            lr = args.lr2 * args.lr_factor
            # lr = args.lr2 
        if "mu_c" in key:
            # lr =  args.lr * args.lr_factor
            lr = args.lr2 * args.lr_factor
            # lr = args.lr2 
        if "mu_sigma" in key:
            # lr =  args.lr * args.lr_factor
            lr = args.lr2 * args.lr_factor
            # lr = args.lr2 
        if "cross" in key:
        #     # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0
            # lr = args.lr2
            # lr = args.lr2 * args.lr_factor
        
        if "avm_mask_head" in key or "avm_id_head" in key:
            # New AVM parameters use the learning rate specified by the guide.
            lr = args.avm_lr

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    return _make_optimizer(args, params)


def build_mask_optimizer(args, model):
    """Optimizer for the mask generator alone (--avm_two_step).

    DPM (paper, implementation details) and DPM++ (make_optimizer_2stage's
    Moptimizer, processor_clipreid_stage3.py) update the mask generator in a
    second step of every iteration. The optimizer type and avm_lr are kept
    from the one-step runs so that only the update scheme changes.
    """
    params = [
        {"params": [value], "lr": args.avm_lr, "weight_decay": args.weight_decay}
        for key, value in model.named_parameters()
        if value.requires_grad and is_mask_generator_param(key)
    ]
    if not params:
        raise ValueError("avm_two_step needs a trainable mask generator")
    return _make_optimizer(args, params)


def _make_optimizer(args, params):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
