from model import objectives
from .avm import (
    FeatureMaskHead,
    HierarchicalMaskGenerator,
    PrototypeHead,
    StaticMask,
    arcface_logits,
    dpm_retrieval_scores,
    feature_gallery_embedding,
    feature_scores,
    mask_statistics,
    occlude_band,
    participation_ratio,
    text_side_masked_scores,
)
from .clip_model import ResidualAttentionBlock, ResidualCrossAttentionBlock, Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice,
            args.img_size,
            args.stride_size,
            download_root=getattr(args, "clip_download_root", None),
        )
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        self.avm_mode = getattr(args, "avm_mode", "none")
        self.avm_margin = getattr(args, "avm_margin", 0.0)
        self.avm_mask_input = getattr(args, "avm_mask_input", "cls")
        self.avm_mask_policy = getattr(args, "avm_mask_policy", "learned")
        self.avm_detach_backbone = getattr(args, "avm_detach_backbone", False)
        self.avm_eval_score = getattr(args, "avm_eval_score", "masked")
        self.avm_occ_ratio = getattr(args, "avm_occ_ratio", 0.0)
        self.avm_occ_weight = getattr(args, "avm_occ_weight", 1.0)
        self.avm_occ_rank_weight = getattr(args, "avm_occ_rank_weight", 0.0)
        self.avm_occ_rank_margin = getattr(args, "avm_occ_rank_margin", 0.05)
        self.avm_id_plain_weight = getattr(args, "avm_id_plain_weight", 0.0)
        self.avm_id_masked_weight = getattr(args, "avm_id_masked_weight", 0.0)
        self.avm_id_margin = getattr(args, "avm_id_margin", 0.5)
        self.avm_id_scale = getattr(args, "avm_id_scale", 30.0)
        self.avm_id_classes = getattr(args, "avm_id_classes", 0)
        self.avm_id_offset = getattr(args, "avm_id_offset", 0)
        self.avm_id_head = None
        self.avm_eff_floor = getattr(args, "avm_eff_floor", 0.0)
        self.avm_eff_floor_weight = getattr(args, "avm_eff_floor_weight", 1.0)

        if self.avm_mode != "dpm" and (
            self.avm_margin != 0.0
            or self.avm_mask_input != "cls"
            or self.avm_mask_policy != "learned"
            or self.avm_detach_backbone
            or self.avm_occ_ratio != 0.0
            or self.avm_id_plain_weight != 0.0
            or self.avm_id_masked_weight != 0.0
            or self.avm_eff_floor != 0.0
        ):
            raise ValueError(
                "--avm_margin / --avm_mask_input / --avm_mask_policy / "
                "--avm_detach_backbone / --avm_occ_* / --avm_id_* / "
                "--avm_eff_floor only apply to --avm_mode dpm"
            )

        if self.avm_mode == "feature":
            self.avm_mask_head = FeatureMaskHead(self.embed_dim)
        elif self.avm_mode == "dpm":
            self._build_dpm_mask_head()
        elif self.avm_mode == "none":
            self.avm_mask_head = None
        else:
            raise ValueError(f"Unsupported AVM mode: {self.avm_mode}")

        if 'fta' in args.loss_names:  
            self.num_query = 4
            self.query = nn.Parameter(torch.randn(self.num_query, self.embed_dim))
            
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            self.mlp_logsigma2 = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim*2),
                nn.ReLU(),
                nn.Linear(self.embed_dim*2, self.embed_dim)
            )

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def _build_dpm_mask_head(self):
        if self.avm_margin < 0:
            raise ValueError("avm_margin must be non-negative")
        if self.avm_eval_score not in ("masked", "plain", "sum"):
            raise ValueError(f"Unsupported avm_eval_score: {self.avm_eval_score}")

        policy, source = self.avm_mask_policy, self.avm_mask_input
        if policy == "learned":
            if source == "hmg":
                visual = self.base_model.visual
                self.avm_mask_head = HierarchicalMaskGenerator(
                    width=visual.transformer.width,
                    out_dim=self.embed_dim,
                    grid_hw=(visual.num_y, visual.num_x),
                )
            elif source == "cls":
                self.avm_mask_head = FeatureMaskHead(self.embed_dim)
            else:
                raise ValueError(f"Unsupported avm_mask_input: {source}")
        elif source != "cls":
            raise ValueError(
                f"avm_mask_policy={policy} takes no image input; "
                "leave avm_mask_input at cls"
            )
        elif policy == "static":
            self.avm_mask_head = StaticMask(self.embed_dim)
        elif policy == "ones":
            if self.avm_detach_backbone:
                raise ValueError(
                    "avm_mask_policy=ones with avm_detach_backbone leaves "
                    "the masked branch with nothing to train"
                )
            self.avm_mask_head = None
        else:
            raise ValueError(f"Unsupported avm_mask_policy: {policy}")

        if self.avm_occ_ratio:
            if not 0.0 < self.avm_occ_ratio < 1.0:
                raise ValueError("avm_occ_ratio must be in (0, 1)")
            if policy != "learned":
                raise ValueError(
                    "controlled occlusion needs an image-dependent mask "
                    "(avm_mask_policy=learned)"
                )
            if min(self.avm_occ_weight, self.avm_occ_rank_weight, self.avm_occ_rank_margin) < 0:
                raise ValueError("avm_occ_* weights and margin must be non-negative")
            if self.avm_occ_weight == 0 and self.avm_occ_rank_weight == 0:
                raise ValueError("avm_occ_ratio is set but no loss uses the occluded copy")

        if min(self.avm_id_plain_weight, self.avm_id_masked_weight) < 0:
            raise ValueError("avm_id_* weights must be non-negative")
        if self.avm_id_plain_weight > 0 or self.avm_id_masked_weight > 0:
            if self.avm_id_classes <= 0:
                raise ValueError(
                    "identity losses need avm_id_classes (finetune.py sets it "
                    "to max train pid + 1)"
                )
            self.avm_id_head = PrototypeHead(self.avm_id_classes, self.embed_dim)

        if self.avm_eff_floor:
            if not 0.0 < self.avm_eff_floor < 1.0:
                raise ValueError("avm_eff_floor must be in (0, 1)")
            if self.avm_eff_floor_weight <= 0:
                raise ValueError("avm_eff_floor_weight must be positive")
            if policy == "ones":
                raise ValueError("an all-ones mask always has eff = 1; the floor does nothing")

    def _dpm_mask(self, aerial_cls, hidden):
        """Aerial-conditioned channel mask [B, embed_dim] for the dpm mode."""
        if self.avm_mask_policy == "ones":
            return torch.ones_like(aerial_cls, dtype=torch.float32)
        if self.avm_mask_policy == "static":
            return self.avm_mask_head(aerial_cls.shape[0])
        if self.avm_mask_input == "hmg":
            return self.avm_mask_head(hidden)
        return self.avm_mask_head(aerial_cls)

    def _encode_aerial_with_hidden(self, image):
        return self.base_model.visual.forward_with_hidden(
            image.type(self.base_model.dtype),
            HierarchicalMaskGenerator.LAYERS,
        )

    def _dpm_id_losses(self, ret, i_feats, t_feats, aerial_in, avm_mask, pids):
        """Idea 2: DPM's plain + masked identity losses (loss/make_loss.py)."""
        # AERI-PEDES train pids are int(anno['pid']) - 1 and can be -1, so
        # shift them to 0-based class indices (finetune.py sets the offset).
        labels = pids.long() + self.avm_id_offset
        num_ids = self.avm_id_head.weight.shape[0]
        low, high = int(labels.min()), int(labels.max())
        if low < 0 or high >= num_ids:
            raise RuntimeError(
                f"identity labels span [{low}, {high}] after offset "
                f"{self.avm_id_offset}; avm_id_classes is {num_ids}"
            )
        if self.avm_id_plain_weight > 0:
            # Plain branch: shared prototypes for aerial and text, so the
            # prototype space is the space the mask meets text in at test.
            ret["avm_id_loss"] = self.avm_id_plain_weight * objectives.compute_id(
                self.avm_id_head.plain_logits(i_feats),
                self.avm_id_head.plain_logits(t_feats),
                labels,
            )
        if self.avm_id_masked_weight > 0:
            logits = arcface_logits(
                self.avm_id_head.masked_cosine(aerial_in, avm_mask),
                labels,
                scale=self.avm_id_scale,
                margin=self.avm_id_margin,
            )
            ret["avm_mid_loss"] = self.avm_id_masked_weight * F.cross_entropy(logits, labels)

    def _dpm_occlusion_losses(self, ret, images, aerial_in, text_in, hidden_in, avm_mask, pids, logit_scale):
        """Idea 1: an occluded aerial copy supervises the mask instead of q_k.

        The occluded copy goes through the backbone without gradients and
        the clean side is detached, so these losses only train the mask
        generator.
        """
        with torch.no_grad():
            occluded = occlude_band(images, self.avm_occ_ratio)
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                if self._uses_hidden_states():
                    occ_feats, occ_hidden = self._encode_aerial_with_hidden(occluded)
                else:
                    occ_feats, occ_hidden = self.base_model.encode_image(occluded), None
        occ_cls = occ_feats[:, 0, :].float()
        occ_mask = self._dpm_mask(occ_cls, occ_hidden)
        occ_eff = participation_ratio(occ_mask)
        ret["avm_occ_eff"] = occ_eff.detach().mean()

        if self.avm_occ_weight > 0:
            # The masked metric must still find the right text when part of
            # the person is missing, with the same margin as the clean branch.
            occ_scores = text_side_masked_scores(text_in.detach(), occ_cls, occ_mask)
            ret["avm_occ_loss"] = self.avm_occ_weight * objectives.compute_sdm_from_scores(
                raw_scores_t2i=occ_scores,
                pid=pids,
                logit_scale=logit_scale,
                margin=self.avm_margin,
            )
        if self.avm_occ_rank_weight > 0:
            # Seeing less should mean trusting fewer channels. The clean mask
            # is re-predicted from detached inputs unless they already are.
            if self.avm_detach_backbone:
                clean_mask = avm_mask
            else:
                clean_hidden = None
                if hidden_in is not None:
                    clean_hidden = {k: v.detach() for k, v in hidden_in.items()}
                clean_mask = self._dpm_mask(aerial_in.detach(), clean_hidden)
            clean_eff = participation_ratio(clean_mask)
            ret["avm_occ_rank_loss"] = self.avm_occ_rank_weight * F.relu(
                occ_eff - clean_eff + self.avm_occ_rank_margin
            ).mean()

    def _uses_hidden_states(self):
        return (
            self.avm_mode == "dpm"
            and self.avm_mask_policy == "learned"
            and self.avm_mask_input == "hmg"
        )

    def retrieval_scores(self, qfeats, gfeats):
        """Named [N_text, N_aerial] score matrices used by the evaluator."""
        if self.avm_mode != "dpm":
            raise RuntimeError("retrieval_scores is only defined for avm_mode=dpm")
        return dpm_retrieval_scores(qfeats, gfeats, self.embed_dim)

    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x, None)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        hidden = None
        if self._uses_hidden_states():
            image_feats, hidden = self._encode_aerial_with_hidden(image)
        else:
            image_feats = self.base_model.encode_image(image)
        aerial_cls = image_feats[:, 0, :].float()

        if self.avm_mode == "feature":
            mask = self.avm_mask_head(aerial_cls)
            return feature_gallery_embedding(aerial_cls, mask)

        if self.avm_mode == "dpm":
            # Gallery row = [aerial CLS | mask]; retrieval_scores unpacks it.
            mask = self._dpm_mask(aerial_cls, hidden)
            return torch.cat([aerial_cls, mask.float()], dim=-1)

        return aerial_cls

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
    def compute_fuzzy_membership(self, A, B):  # compute_fuzzy_membership v3

        A = A.half() # [B，K，D]
        B = B.half() # [B, D]

        Qn_norm  = F.normalize(A.half(), dim=-1)            # [B, K, D]
        Tn_norm  = F.normalize(B.half(), dim=-1) # [B, D]

        log_sigma2 = self.mlp_logsigma2(Tn_norm) 
        sigma2 = torch.exp(log_sigma2)
        sigma2 = torch.clamp(sigma2, min=1e-6)

        Qn_exp = Qn_norm.unsqueeze(2)  
        Tn_exp = Tn_norm.unsqueeze(0).unsqueeze(0)  
        r_dim = Qn_exp * Tn_exp 

        sigma2_exp = sigma2.unsqueeze(0).unsqueeze(0)
        mu_dim = torch.exp(- ((1.0 - r_dim) ** 2) / (2 * sigma2_exp ** 2))  # [B, K, B, D]
        mu_mean = mu_dim.mean(dim=-1)  # [Bq, K, Bt]
        membership = mu_mean.transpose(1, 2).contiguous()  # [B, B, K]

        return membership
    
    def entropy_maximize_loss_from_mu(self, member, dim=-1, eps=1e-8):
        # mu: [Bq, Bt, K]
        member = member.clamp(min=0.0)
        p = member / (member.sum(dim=dim, keepdim=True) + eps)  # p is prob over Bt
        entropy = - (p * torch.log(p + eps)).sum(dim=dim)  # shape [Bq, K]
        loss = - entropy.mean()  # minimize loss => maximize mean entropy
        return loss

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        ground_images = batch['ground_imgs']
        # ground_images = None
        caption_ids = batch['caption_ids']
        hidden = None
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            if self._uses_hidden_states():
                # Same calls as CLIP.forward's single-text branch, with the
                # aerial encoder also returning the blocks the HMG needs.
                if caption_ids.size(0) == 2 * images.size(0):
                    raise RuntimeError(
                        "avm_mask_input=hmg does not support the doubled-text "
                        "branch of CLIP.forward"
                    )
                image_feats, hidden = self._encode_aerial_with_hidden(images)
                if ground_images is not None:
                    ground_image_feats = self.base_model.encode_image(ground_images)
                else:
                    ground_image_feats = None
                text_feats = self.base_model.encode_text(caption_ids)
            else:
                image_feats, ground_image_feats, text_feats = self.base_model(images, ground_images, caption_ids)

        i_feats = image_feats[:, 0, :].float()
        if ground_image_feats is not None:
            g_i_feats = ground_image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale

        if 'cda' in self.current_task:
            ret.update({'cda_loss': objectives.compute_selective_align_loss(i_feats, g_i_feats, t_feats, batch['pids'], logit_scale)})

        if self.avm_mode == "feature":
            avm_mask = self.avm_mask_head(i_feats)
            raw_scores_t2i = feature_scores(
                text_cls=t_feats,
                aerial_cls=i_feats,
                mask=avm_mask,
            )
            avm_ret_loss = objectives.compute_sdm_from_scores(
                raw_scores_t2i=raw_scores_t2i,
                pid=batch["pids"],
                logit_scale=logit_scale,
            )

            ret.update({
                "avm_ret_loss":
                    self.args.avm_loss_weight * avm_ret_loss,
                "avm_mask_mean":
                    avm_mask.detach().mean(),
                "avm_mask_std":
                    avm_mask.detach().std(unbiased=False),
            })

        if self.avm_mode == "dpm":
            # DPM two-branch objective: the CFAN losses above are the plain
            # branch; this is the masked branch, scored with the aerial mask
            # applied to every text candidate and trained with a margin.
            aerial_in, text_in, hidden_in = i_feats, t_feats, hidden
            if self.avm_detach_backbone:
                aerial_in, text_in = i_feats.detach(), t_feats.detach()
                if hidden is not None:
                    hidden_in = {k: v.detach() for k, v in hidden.items()}

            avm_mask = self._dpm_mask(aerial_in, hidden_in)
            masked_scores_t2i = text_side_masked_scores(
                text_feats=text_in,
                aerial_feats=aerial_in,
                mask=avm_mask,
            )
            avm_ret_loss = objectives.compute_sdm_from_scores(
                raw_scores_t2i=masked_scores_t2i,
                pid=batch["pids"],
                logit_scale=logit_scale,
                margin=self.avm_margin,
            )
            instance_std, participation = mask_statistics(avm_mask)

            ret.update({
                "avm_ret_loss":
                    self.args.avm_loss_weight * avm_ret_loss,
                "avm_mask_mean":
                    avm_mask.detach().mean(),
                "avm_mask_std":
                    avm_mask.detach().std(unbiased=False),
                "avm_mask_inst_std": instance_std,
                "avm_mask_eff": participation,
            })

            if self.avm_eff_floor:
                # Batch 5: the more channels the mask dropped, the worse the
                # masked score (eff 0.87 -> -0.7, 0.55 -> -3.8, 0.35 -> -6.8).
                # A hinge floor on the participation ratio caps how selective
                # the mask may become, like DPM++'s budget loss.
                ret["avm_eff_floor_loss"] = self.avm_eff_floor_weight * F.relu(
                    self.avm_eff_floor - participation_ratio(avm_mask)
                ).mean()

            if self.avm_id_head is not None:
                self._dpm_id_losses(ret, i_feats, t_feats, aerial_in, avm_mask, batch["pids"])
            if self.avm_occ_ratio:
                self._dpm_occlusion_losses(
                    ret, images, aerial_in, text_in, hidden_in, avm_mask,
                    batch["pids"], logit_scale,
                )

        if 'fta' in self.current_task:
            B = text_feats.shape[0]
            query_expand = self.query.unsqueeze(0).expand(B, -1, -1)

            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                Q_v = self.cross_former(query_expand.half(), image_feats, image_feats) 
                Q_t = self.cross_former(query_expand.half(), text_feats, text_feats)  # 

            # v2 cross-modal membership
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                mu_t2v = self.compute_fuzzy_membership(Q_t, t_feats)
                mu_v2t = self.compute_fuzzy_membership(Q_v, i_feats)

            Q_t = F.normalize(Q_t, dim=-1) # [b,4,512]
            Q_v = F.normalize(Q_v, dim=-1) # [b,4,512]

            t2v_simi = torch.einsum('bkd,Bkd->bBk', Q_t, Q_v)
            v2t_simi = torch.einsum('bkd,Bkd->bBk', Q_v, Q_t)
            mu_and = mu_t2v * mu_v2t
            S_t2v = (t2v_simi * mu_and).mean(dim=-1)
            S_v2t = (v2t_simi * mu_and).mean(dim=-1) 

            ret.update({'fta_loss':0.5*objectives.compute_fa_loss(S_t2v, S_v2t, batch['pids'], logit_scale)})

        return ret


def build_finetune_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
