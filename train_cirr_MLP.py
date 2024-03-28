import datetime
import shutil
import time
import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from src.test.utils import evaluate
from src.tools.files import json_dump
from src.tools.utils import calculate_model_params
import torch
import torch.nn.functional as F

class WeightingMLP(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(WeightingMLP, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_dim, embedding_dim // 2)
        self.fc2 = torch.nn.Linear(embedding_dim // 2, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    fabric = instantiate(cfg.trainer.fabric)
    fabric.launch()
    fabric.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    if fabric.global_rank == 0:
        json_dump(OmegaConf.to_container(cfg, resolve=True), "hydra.json")

    data = instantiate(cfg.data)
    loader_train = fabric.setup_dataloaders(data.train_dataloader())
    if cfg.val:
        loader_val = fabric.setup_dataloaders(data.val_dataloader())

    model = instantiate(cfg.model)
    calculate_model_params(model)

    weighting_mlp = WeightingMLP(model.text_proj.out_features).to(fabric.device)

    optimizer = instantiate(
        cfg.model.optimizer,
        params=list(model.parameters()) + list(weighting_mlp.parameters()),
        _partial_=False,
    )

    model, optimizer = fabric.setup(model, optimizer)
    scheduler = instantiate(cfg.model.scheduler)

    fabric.print("Start training")
    start_time = time.time()

    for epoch in range(cfg.trainer.max_epochs):
        scheduler(optimizer, epoch)
        columns = shutil.get_terminal_size().columns
        fabric.print("-" * columns)
        fabric.print(f"Epoch {epoch + 1}/{cfg.trainer.max_epochs}".center(columns))
        train(model, weighting_mlp, loader_train, optimizer, fabric, epoch, cfg)

        if cfg.val:
            fabric.print("Evaluate")
            evaluate(model, loader_val, fabric=fabric)

        state = {
            "epoch": epoch,
            "model": model,
            "weighting_mlp": weighting_mlp,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }

        if cfg.trainer.save_ckpt == "all":
            fabric.save(f"ckpt_{epoch}.ckpt", state)
        elif cfg.trainer.save_ckpt == "last" and epoch == cfg.trainer.max_epochs - 1:
            fabric.save("ckpt_last.ckpt", state)

        fabric.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    fabric.print(f"Training time {total_time_str}")

    for dataset in cfg.test:
        columns = shutil.get_terminal_size().columns
        fabric.print("-" * columns)
        fabric.print(f"Testing on {cfg.test[dataset].dataname}".center(columns))
        data = instantiate(cfg.test[dataset])
        test_loader = fabric.setup_dataloaders(data.test_dataloader())
        test = instantiate(cfg.test[dataset].test)
        test(model, weighting_mlp, test_loader, fabric=fabric)

    fabric.logger.finalize("success")
    fabric.print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def train(model, weighting_mlp, train_loader, optimizer, fabric, epoch, cfg):
    model.train()
    weighting_mlp.train()

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        ref_img, caption, pair_id, *_ = batch
        device = ref_img.device

        ref_img_embs = model.visual_encoder(ref_img)
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        text = model.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        text_embs = model.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        ).last_hidden_state[:, 0, :]

        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = model.tokenizer.enc_token_id
        query_embs = model.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        query_feat = query_embs.last_hidden_state[:, 0, :]

        avg_embs = (query_feat + ref_img_embs[:, 0, :] + text_embs) / 3

        weights = weighting_mlp(avg_embs)
        weighted_embs = weights[:, 0:1] * query_feat + weights[:, 1:2] * ref_img_embs[:, 0, :] + weights[:, 2:3] * text_embs
        weighted_embs = F.normalize(model.text_proj(weighted_embs), dim=-1)

        loss = model(batch, fabric, weighted_embs=weighted_embs)
        fabric.backward(loss)
        optimizer.step()

        if batch_idx % cfg.trainer.print_interval == 0:
            fabric.print(
                f"[{100.0 * batch_idx / len(train_loader):.0f}%]\tLoss: {loss.item():.6f}"
            )

        if batch_idx % cfg.trainer.log_interval == 0:
            fabric.log_dict(
                {
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )

if __name__ == "__main__":
    main()