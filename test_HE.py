import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from models.he_model import HEModel
from dataloaders_for_HE.he_test_datasets.get_HEVPR_test_datasets import get_test_HEVPR_datasets


if __name__ == '__main__':
    dino_feat_dim = {
        'dinov2_vitl14': 1024,
        'dinov2_vitb14': 768,
        'dinov2_vits14': 384,
    }
    dino_name = 'dinov2_vitb14'
    model = HEModel(
        # ---- Encoder
        backbone_arch='mona_vit',
        backbone_config={
            'dino_name': dino_name,
        },
        agg_arch='g_dino',
        agg_config={'p': 3},
    )

    ckpt_path = './weights/HE_Mona/HEVPR_best.ckpt'  # best weights
    state_dict = torch.load(ckpt_path)
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    elif 'model_state_dict' in state_dict:
        missing_keys = model.load_state_dict(state_dict['model_state_dict'], strict=False)
        print("Missing keys:", missing_keys)
    else:
        print(f"Using the raw model from {ckpt_path}, Loading")
        # modified for CricaVPR
        defined_model_keys = list(model.state_dict().keys())
        defined_model_params = list(model.state_dict().values())
        for pretrained_key_index, pretrained_key in enumerate(state_dict.items()):
            to_load_key = defined_model_keys[pretrained_key_index]
            to_load_param_shape = defined_model_params[pretrained_key_index].shape
            if to_load_param_shape == pretrained_key[1].shape:
                model.state_dict()[to_load_key] = state_dict[pretrained_key[0]]
            else:
                raise Exception('pretrained params shape is wrong')
    model.eval()
    print(f"Loaded model from {ckpt_path} Successfully!")

    resize = [224, 224]
    test_datasets_batch = get_test_HEVPR_datasets(resize)

    for test_dataset in [test_datasets_batch[1], test_datasets_batch[3]]:
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        trainer_tmp = pl.Trainer(max_epochs=1, accelerator="auto", devices=1)
        trainer_tmp.test(model=model, dataloaders=test_loader)
        del trainer_tmp, test_loader
        model.test_pred_feat = []
