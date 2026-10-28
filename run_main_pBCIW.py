from models.eegpt import EEGTransformer
import models.labram
from models.cbramod import CBraMod
from models.codebrain import SSSM as CodeBrain
from models.biot import BIOTEncoder
from models.bendr import ConvEncoderBENDR
from models.singlem import Config, EEGEncoder
# from models.cbramod import CBraMod
# from models.labram import LaBraM
# from models.codebrain import CodeBrain

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import LinearWithConstraint, Conv1dWithConstraint
from eval import get_metrics
from dataset.dataset import get_data_matb
import math
from timm.models import create_model

from functools import partial
from pytorch_lightning import loggers as pl_loggers


use_channels_names_full = [
    # ===== Frontal pole / Anterior frontal =====
    'FP1', 'FPZ', 'FP2',
    'AF3',        'AF4',
    # ===== Frontal =====
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    # ===== Fronto-temporal / Fronto-central =====
    'FT7',
    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'FT8',
    # ===== Temporal / Central =====
    'T7',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'T8',
    # ===== Temporo-parietal / Centro-parietal =====
    'TP7',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8',
    # ===== Parietal =====
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    # ===== Parieto-occipital =====
    'PO7',  'PO3', 'POZ', 'PO4',  'PO8',
    # ===== Occipital / Cerebellar =====
    'O1', 'OZ', 'O2',
]
use_channels_names = [ch for ch in use_channels_names_full if ch.upper() not in {'FPZ', 'CZ'}]


class LitModelCausal(pl.LightningModule):

    def __init__(
        self,
        fold,
        model_choice='EEGPT',
        dataset='bcic2a',
        load_path=None,
        freeze_backbone=True,
        is_load_pretrained=True,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.is_load_pretrained = is_load_pretrained

        self.fold = fold
        self.model_choice = model_choice.upper()
        self.dataset = dataset
        self.chans_num = 22

        # ==========================================================
        # 1. 初始化基础模型
        # ==========================================================
        self._build_backbone(load_path)

        # ==========================================================
        # 2. 初始化下游分类头
        #    不同基础模型如果输出维度不同，在这里分别定义
        # ==========================================================
        self._build_classifier()

        self.drop = nn.Dropout(p=0.50)

        self.loss_fn = nn.CrossEntropyLoss()

        self.running_scores = {
            "train": [],
            "valid": [],
            "test": []
        }

        self.is_sanity = True
        self.best_epoch = -1
        self.best_metrics = {}

    # ==============================================================
    # Backbone initialization
    # ==============================================================
    def _build_backbone(self, load_path):

        # ----------------------------------------------------------
        # EEGPT
        # ----------------------------------------------------------
        if self.model_choice == 'EEGPT':

            if load_path is None:
                load_path = (
                    "/home/sy/code/EEGPT/EEGPT-main/"
                    "downtasks/logs/EEGPT/checkpoint/"
                    "eegpt_mcae_58chs_4s_large4E.ckpt"
                )

            self.backbone = EEGTransformer(
                img_size=[56, 512],
                patch_size=32 * 2,
                embed_num=4,
                embed_dim=512,
                depth=8,
                num_heads=8,
                mlp_ratio=4.0,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                init_std=0.02,
                qkv_bias=True,
                norm_layer=partial(
                    nn.LayerNorm,
                    eps=1e-6
                )
            )

            # EEGPT需要通道ID
            self.chans_id = self.backbone.prepare_chan_ids(use_channels_names)

            # 加载预训练参数
            if self.is_load_pretrained:
                pretrain_ckpt = torch.load(load_path, map_location='cpu', weights_only=False)

                backbone_state = {}

                for k, v in pretrain_ckpt['state_dict'].items():
                    if k.startswith("target_encoder."):
                        backbone_state[k[15:]] = v

                msg = self.backbone.load_state_dict(backbone_state, strict=True)

                print(
                    f"[EEGPT] Loaded pretrained checkpoint: {load_path}"
                )
                print(msg)

            # 用于构建分类头
            self.feature_dim = 1024
            self.feature_num = 16

        # ----------------------------------------------------------
        # CBraMod
        # ----------------------------------------------------------
        elif self.model_choice == 'CBRAMOD':
            self.backbone = CBraMod(
                in_dim=200, out_dim=200, d_model=200,
                dim_feedforward=800, seq_len=1,
                n_layer=12, nhead=8
            )
            if self.is_load_pretrained:
                state_dict = torch.load(load_path, map_location="cpu", weights_only=False)
                self.backbone.load_state_dict(state_dict)

        # ----------------------------------------------------------
        # LaBraM
        # ----------------------------------------------------------
        elif self.model_choice == 'LABRAM':
            self.backbone = create_model("labram_base_patch200_200", 
                                # checkpoint_path= ,
                                qkv_bias=False,
                                rel_pos_bias=True,
                                num_classes=4,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                attn_drop_rate=0.0,
                                drop_block_rate=None,
                                use_mean_pooling=True,
                                init_scale=0.001,
                                use_rel_pos_bias=True,
                                use_abs_pos_emb=True,
                                init_values=0.1,)
            if self.is_load_pretrained:
                checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
                new_checkpoint = {}
                for k,v in checkpoint['model'].items():
                    if k.startswith('student.'):
                        new_checkpoint[k[len('student.'):]] = v
                self.backbone.load_state_dict(new_checkpoint, strict=False)
            self.backbone.proj_out = nn.Identity()

        # ----------------------------------------------------------
        # CodeBrain
        # ----------------------------------------------------------
        elif self.model_choice == 'CODEBRAIN':
            self.backbone = CodeBrain(
                in_channels = 200, res_channels = 200,
                skip_channels = 200, out_channels = 200,
                num_res_layers = 8,
                diffusion_step_embed_dim_in = 200,
                diffusion_step_embed_dim_mid = 200,
                diffusion_step_embed_dim_out = 200,
                s4_lmax = 570,
                s4_d_state = 64,
                s4_dropout = 0.1,
                s4_bidirectional = True,
                s4_layernorm = True,
                codebook_size_t = 4096,
                codebook_size_f = 4096,
                if_codebook = False)
            if self.is_load_pretrained:
                state_dict = torch.load(load_path, map_location='cpu', weights_only=False)
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k[7:]] = v
                self.backbone.load_state_dict(new_state_dict)
            self.backbone.proj_out = nn.Identity()

        elif self.model_choice == 'BIOT':
            self.backbone = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=16, n_fft=200, hop_length=100)
            if self.is_load_pretrained:
                self.backbone.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=False))

        elif self.model_choice == 'BENDR':
            self.backbone = ConvEncoderBENDR(20, encoder_h=512, dropout=0., projection_head=False)
            self.scale_param    = torch.nn.Parameter(torch.tensor(1.))
            self.channel_conv     = torch.nn.Conv1d(61, 19, 1)
            if self.is_load_pretrained:
                self.backbone.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=False))

        elif self.model_choice == 'SINGLEM':
            config = Config()
            if self.is_load_pretrained:
                    payload = torch.load(load_path, map_location='cpu', weights_only=True,)
                    for name, value in payload["model_config"].items():
                        setattr(config, name, value)
            config.mask_prob = 0.0
            self.backbone = EEGEncoder(config)

            if self.is_load_pretrained:
                self.backbone.load_state_dict(payload["encoder_state_dict"])
                
        # ----------------------------------------------------------
        # Other models
        # ----------------------------------------------------------
        else:
            raise ValueError(
                f"Unsupported model_choice: {self.model_choice}. "
                f"Supported models: "
                f"EEGPT / CBraMod / LaBraM / CodeBrain / BIOT"
            )

        # ----------------------------------------------------------
        # 冻结基础模型
        # ----------------------------------------------------------
        # for blk in model.blocks:
        #     for p in blk.parameters():
        #         p.requires_grad = False
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    # ==============================================================
    # Classifier initialization
    # ==============================================================
    def _build_classifier(self):

        # ----------------------------------------------------------
        # EEGPT
        # ----------------------------------------------------------
        if self.model_choice == 'EEGPT':

            self.linear_probe1 = LinearWithConstraint(2048,16,max_norm=1)
            self.linear_probe2 = LinearWithConstraint(8 * 16,3,max_norm=0.25)

        # ----------------------------------------------------------
        # CBraMod
        # ----------------------------------------------------------
        elif self.model_choice == 'CBRAMOD':
            self.linear_probe1 = nn.Linear(61*200*3, 200)
            self.linear_probe2 = nn.Linear(200, 4)
            self.elu = nn.ELU()


        # ----------------------------------------------------------
        # LaBraM
        # ----------------------------------------------------------
        elif self.model_choice == 'LABRAM':
            self.linear_probe1 = LinearWithConstraint(200, 4, max_norm=1)

        # ----------------------------------------------------------
        # CodeBrain
        # ----------------------------------------------------------
        elif self.model_choice == 'CODEBRAIN':
            self.linear_probe1 = nn.Linear(61*200, 200)
            self.linear_probe2 = nn.Linear(200, 4)
            self.elu = nn.ELU()
            self.codebrain_dropout = nn.Dropout(p=0.3)

        elif self.model_choice == 'BIOT':
            self.chan_conv = Conv1dWithConstraint(61, 16, 1, max_norm=1)
            self.linear_probe1 = LinearWithConstraint(256, 4, max_norm=1)
            self.elu = nn.ELU()

        elif self.model_choice == 'BENDR':
            self.linear_probe1   = torch.nn.Linear(3584, 64)
            self.linear_probe2   = torch.nn.Linear(64, 4)

        elif self.model_choice == 'SINGLEM':
            self.linear_probe1 = nn.Linear(56 * 4 * 16, 200)
            self.linear_probe2 = nn.Linear(200, 4)
            self.elu = nn.ELU()
            self.singlem_dropout = nn.Dropout(p=0.3)

    # ==============================================================
    # Forward
    # ==============================================================
    def forward(self, x):

        if self.freeze_backbone:
            self.backbone.eval()

        if self.model_choice == 'EEGPT':
            h = self.backbone(x, self.chans_id.to(x.device))
            h = h.flatten(2)
            h = self.linear_probe1(self.drop(h))
            h = h.flatten(1)
            logit = self.linear_probe2(h)

        # ----------------------------------------------------------
        # CBraMod
        # ----------------------------------------------------------
        elif self.model_choice == 'CBRAMOD':
            B, C, T = x.shape
            N = T // 200
            x = x[:,:,:N*200]
            h = self.backbone(x)
            h = h.flatten(1)
            h = self.elu(self.linear_probe1(h))
            h = self.drop(h)
            logit = self.linear_probe2(h)

        # ----------------------------------------------------------
        # LaBraM
        # ----------------------------------------------------------
        elif self.model_choice == 'LABRAM':
            B, C, T = x.shape
            if T%200!=0: 
                x = x[:,:,0:T-T%200]
                T = T-T%200
            x = x.reshape((B,C,T//200,200))
            h = self.backbone.forward_features(x, input_chans=[i for i in range(C + 1)], return_all_tokens=False)
            h = self.linear_probe1(h.flatten(1))
            logit = self.drop(h)

        # ----------------------------------------------------------
        # CodeBrain
        # ----------------------------------------------------------
        elif self.model_choice == 'CODEBRAIN':
            bs, channel, seq_len = x.shape
            x = x[:,:,:seq_len//200*200]
            x = x.contiguous().view(bs, channel, seq_len//200, 200)
            h = self.backbone(x)
            h = h.mean(dim=2)
            h = h.flatten(1)
            h = self.elu(self.linear_probe1(h))
            h = self.codebrain_dropout(h)
            logit = self.linear_probe2(h)

        # ----------------------------------------------------------
        # BIOT
        # ----------------------------------------------------------
        elif self.model_choice == 'BIOT':
            h = self.chan_conv(x)
            h = self.backbone(h)
            h = self.elu(h)
            logit = self.linear_probe1(h)

        elif self.model_choice == 'BENDR':
            x = self.channel_conv(x)
            x = torch.cat([x, self.scale_param.repeat((x.shape[0], 1, x.shape[-1]))], dim=-2)
            h = self.backbone(x)
            h = h.flatten(1)
            h = self.linear_probe1(h)
            h = self.drop(h)
            logit = self.linear_probe2(h)

        elif self.model_choice == 'SINGLEM':
            B, C, T = x.shape

            assert T % 128 == 0
            N = T // 128
            x = x.reshape(B, C, N, 128)

            x = x.reshape(B * C, N, 128)
            h, _, _ = self.backbone(x)
            h = h.reshape(B, C, N, 16)
            h = h.reshape(B, C * N * 16)

            h = self.singlem_dropout(self.elu(self.linear_probe1(h)))
            logit = self.linear_probe2(h)

        return x, logit

    # ==============================================================
    # Training
    # ==============================================================
    def training_step(self, batch, batch_idx):

        x, y = batch

        # CrossEntropyLoss直接使用整数标签即可
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)

        accuracy = ((torch.argmax(logit, dim=-1) == label).float().mean())

        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)

        return loss

    # ==============================================================
    # Validation
    # ==============================================================
    def on_validation_epoch_start(self):

        self.running_scores["valid"] = []

        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):

        x, y = batch

        label = y.long()

        x, logit = self.forward(x)

        loss = self.loss_fn(logit, label)

        accuracy = ((torch.argmax(logit, dim=-1) == label).float().mean())

        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log( 'valid_acc', accuracy, on_epoch=True, on_step=False)
        self.running_scores["valid"].append(
            (
                label.detach().cpu(),
                logit.detach().cpu()
            )
        )

        return loss

    def on_validation_epoch_end(self):

        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()

        labels = []
        y_scores = []

        for label, score in self.running_scores["valid"]:
            labels.append(label)
            y_scores.append(score)

        labels = torch.cat(labels, dim=0)

        y_scores = torch.cat(y_scores, dim=0)

        metrics = [
            "accuracy",
            "balanced_accuracy",
            "cohen_kappa",
            "f1_weighted",
            "f1_macro",
            "f1_micro"
        ]

        results = get_metrics(
            y_scores.cpu().numpy(),
            labels.cpu().numpy(),
            metrics,
            False
        )

        for key, value in results.items():
            self.log('valid_' + key, value, on_epoch=True, on_step=False, sync_dist=True)

        if (self.best_epoch == -1 or results['balanced_accuracy'] > self.best_metrics.get('balanced_accuracy',0)):
            self.best_epoch = self.current_epoch
            self.best_metrics = results

        return super().on_validation_epoch_end()

    # ==============================================================
    # End
    # ==============================================================
    def on_fit_end(self):

        print(
            f"Best epoch: {self.best_epoch}"
        )

        for metric_name, metric_value in self.best_metrics.items():
            print(
                f"Best val_{metric_name}: "
                f"{metric_value:.4f}"
            )

        # 根据基础模型分别保存结果
        if self.freeze_backbone:
            freeze_str = "freeze"
        elif self.is_load_pretrained:
            freeze_str = "unfreeze"
        else:
            freeze_str = "unfreeze_random"
        log_path = (
            f'./logs/pBCIW/'
            f'{self.model_choice}_{self.dataset}_{freeze_str}.txt'
        )

        with open(log_path, 'a') as f:

            kv_val = " ".join(
                [
                    f"{k}={v:.4f}"
                    for k, v
                    in self.best_metrics.items()
                ]
            )

            f.write(
                f"Fold {self.fold}\t"
                f"BestEpoch={self.best_epoch}\t"
                f"Valid: {kv_val}\n"
            )

    # ==============================================================
    # Optimizer
    # ==============================================================
    def configure_optimizers(self):

        trainable_parameters = [
            p for p in self.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            weight_decay=0.01
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            pct_start=0.2
        )

        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': None,
        }

        return (
            {
                'optimizer': optimizer,
                'lr_scheduler': lr_dict
            },
        )

if __name__ == "__main__":
    for i in range(1, 16):
        batch_size = 64
        data_path = "/home/sy/data/downtasks/PBCIW/Sub_S1_single"

        model_choice = 'CODEBRAIN'
        dataset = 'pBCIW'

        if model_choice == 'EEGPT' or model_choice == 'SINGLEM':
            train_dataset,valid_dataset,test_dataset = get_data_matb(i, data_path, use_channels=use_channels_names, target_length=512)
        else:
            train_dataset,valid_dataset,test_dataset = get_data_matb(i, data_path, target_length=600)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=0, shuffle=False)

        global max_epochs
        global steps_per_epoch
        global max_lr

        max_lr = 5e-4
        max_epochs = 100
        steps_per_epoch = math.ceil(len(train_loader))

        load_path_eegpt = "./checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt"
        load_path_labram = "./checkpoints/labram-base.pth"
        load_path_cbramod = "./checkpoints/cbramod.pth"
        load_path_codebrain = "./checkpoints/codebrain.pth"
        load_path_biot = "./checkpoints/biot_prest_16_channels.ckpt"
        load_path_bendr = "./checkpoints/bendr.pth"
        load_path_singlem = "./checkpoints/singlem.pt"
        if model_choice == 'EEGPT':
            load_path = load_path_eegpt
        elif model_choice == 'LABRAM':
            load_path = load_path_labram
        elif model_choice == 'CBRAMOD':
            load_path = load_path_cbramod
        elif model_choice == 'CODEBRAIN':
            load_path = load_path_codebrain
        elif model_choice == 'BIOT':
            load_path = load_path_biot
        elif model_choice == 'BENDR':
            load_path = load_path_bendr
        elif model_choice == 'SINGLEM':
            load_path = load_path_singlem

        # Three Cases: 1.(True, True) 2.(False, True) 3.(False, False)
        litmodel = LitModelCausal(
            fold=i,
            model_choice=model_choice,
            dataset=dataset,
            load_path=load_path,
            freeze_backbone=False,
            is_load_pretrained=True,
        )

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        callbacks = [lr_monitor]
        
        trainer = pl.Trainer(accelerator='cuda',
                            devices=[0,], 
                            max_epochs=max_epochs, 
                            callbacks=callbacks,
                            enable_checkpointing=True,
                            num_sanity_val_steps=0,
                            logger=[pl_loggers.TensorBoardLogger('./logs/pBCIW/', name=f"{model_choice}_{dataset}_2_tb", version=f"subject{i}_FT"), 
                                    pl_loggers.CSVLogger('./logs/pBCIW/', name=f"{model_choice}_{dataset}_2_csv", version=f"subject{i}_FT")])

        trainer.fit(litmodel, train_loader, test_loader)
        # trainer.test(litmodel, test_loader)