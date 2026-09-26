import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_save", default=False, action='store_true',
                        help="skip checkpoint saving (for ablation sweeps)")

    parser.add_argument("--finetune", type=str, default="pretrain/HAMbest0.pth")
    parser.add_argument("--clip_download_root", type=str, default=None,
                        help="directory for downloading and caching OpenAI CLIP weights")
    parser.add_argument("--pretrain", type=str, default="")
    parser.add_argument("--nam", default=False, action='store_true')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=True, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")

    ######################## AVM settings ########################
    parser.add_argument(
        "--avm_mode",
        type=str,
        default="none",
        choices=["none", "feature", "dpm"],
        help="aerial visibility masking mode",
    )
    parser.add_argument(
        "--avm_margin",
        type=float,
        default=0.0,
        help="dpm: additive cosine margin on same-PID pairs in the masked SDM",
    )
    parser.add_argument(
        "--avm_mask_input",
        type=str,
        default="cls",
        choices=["cls", "hmg"],
        help="dpm: mask from the aerial CLS (MLP) or DPM++ hierarchical "
             "mask generator on blocks 2/4/10/12",
    )
    parser.add_argument(
        "--avm_mask_policy",
        type=str,
        default="learned",
        choices=["learned", "static", "ones"],
        help="dpm: per-image learned mask, one shared learned mask, "
             "or all ones (margin-only control)",
    )
    parser.add_argument(
        "--avm_detach_backbone",
        default=False,
        action='store_true',
        help="dpm: masked-branch loss updates only the mask generator",
    )
    parser.add_argument(
        "--avm_eval_score",
        type=str,
        default="masked",
        choices=["masked", "plain", "sum"],
        help="dpm: score reported as t2i and used to pick best0",
    )
    # Idea 1: controlled occlusion (dpm, learned mask only).
    parser.add_argument(
        "--avm_occ_ratio",
        type=float,
        default=0.0,
        help="dpm: height fraction blanked in an extra occluded aerial copy "
             "(0 = off); its backbone pass runs without gradients",
    )
    parser.add_argument(
        "--avm_occ_weight",
        type=float,
        default=1.0,
        help="dpm: weight of the masked SDM between texts and the occluded copy",
    )
    parser.add_argument(
        "--avm_occ_rank_weight",
        type=float,
        default=0.0,
        help="dpm: weight of the hinge asking the occluded mask to use fewer "
             "channels than the clean mask",
    )
    parser.add_argument(
        "--avm_occ_rank_margin",
        type=float,
        default=0.05,
        help="dpm: required participation-ratio gap, clean minus occluded",
    )
    # Idea 2: literal DPM masked ID loss (dpm).
    parser.add_argument(
        "--avm_id_plain_weight",
        type=float,
        default=0.0,
        help="dpm: weight of the plain identity softmax on aerial and text",
    )
    parser.add_argument(
        "--avm_id_masked_weight",
        type=float,
        default=0.0,
        help="dpm: weight of the masked ArcFace identity loss on aerial",
    )
    parser.add_argument(
        "--avm_id_margin",
        type=float,
        default=0.5,
        help="dpm: ArcFace angular margin of the masked identity loss",
    )
    parser.add_argument(
        "--avm_id_scale",
        type=float,
        default=30.0,
        help="dpm: ArcFace scale of the masked identity loss",
    )
    parser.add_argument(
        "--avm_id_classes",
        type=int,
        default=0,
        help="dpm: number of identity prototypes; 0 = max - min train pid + 1",
    )
    parser.add_argument(
        "--avm_id_offset",
        type=int,
        default=0,
        help="dpm: added to pids to get 0-based class indices "
             "(finetune.py sets it to -min train pid; AERI-PEDES has pid -1)",
    )
    parser.add_argument(
        "--avm_eff_floor",
        type=float,
        default=0.0,
        help="dpm: hinge floor on the mask participation ratio (0 = off); "
             "limits how many channels the mask may drop",
    )
    parser.add_argument(
        "--avm_eff_floor_weight",
        type=float,
        default=1.0,
        help="dpm: weight of the participation-ratio floor hinge",
    )
    parser.add_argument(
        "--avm_loss_weight",
        type=float,
        default=1.0,
        help="weight of the AVM retrieval loss",
    )
    parser.add_argument(
        "--avm_lr",
        type=float,
        default=1e-4,
        help="learning rate for newly initialized AVM parameters",
    )
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr2", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 40))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="AERI-PEDES", help="[CUHK-PEDES, AERI-PEDES, AGDataAttr]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="/data1/Datasets/ReID/")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args
