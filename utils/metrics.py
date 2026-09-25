from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity.data.cpu(), dim=1, descending=True)
        indices = indices.to(similarity.device)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat.data.cpu())
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat.data.cpu())
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats.cuda(), gfeats.cuda(), qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        if getattr(model, "avm_mode", "none") == "dpm":
            return self._eval_named_scores(model, qfeats, gfeats, qids, gids)
        if qfeats.shape[1] != gfeats.shape[1]:
            # e.g. a dpm model wrapped in DataParallel: packed gallery rows
            # must not be scored with a plain cosine.
            raise ValueError(
                f"text width {qfeats.shape[1]} != image width {gfeats.shape[1]}; "
                "pass the unwrapped model"
            )

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "RSum", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["RSum"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))

        return t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]

    def _eval_named_scores(self, model, qfeats, gfeats, qids, gids):
        """Rank every score the dpm model exposes; report the chosen one.

        The chosen score is printed as the 't2i' row, so log parsers that
        read one '| t2i' row per evaluation keep working; the plain, masked
        and sum rows are named so they never match '^| *t2i'.
        """
        selected = model.avm_eval_score
        table = PrettyTable(["task", "R1", "R5", "R10", "RSum", "mAP", "mINP"])
        rows, selected_rsum = [], None

        for name, similarity in model.retrieval_scores(qfeats, gfeats).items():
            cmc, mAP, mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
            cmc, mAP, mINP = cmc.numpy(), mAP.numpy(), mINP.numpy()
            row = [cmc[0], cmc[4], cmc[9], cmc[0] + cmc[4] + cmc[9], mAP, mINP]
            rows.append([name] + row)
            if name == selected:
                table.add_row(['t2i'] + row)
                selected_rsum = row[3]
            del similarity

        if selected_rsum is None:
            raise ValueError(f"avm_eval_score {selected!r} is not one of the model's scores")

        for row in rows:
            table.add_row(row)
        for column in ["R1", "R5", "R10", "RSum", "mAP", "mINP"]:
            table.custom_format[column] = lambda f, v: f"{v:.3f}"
        self.logger.info(f'dpm scores, t2i = {selected}\n' + str(table))

        return selected_rsum
