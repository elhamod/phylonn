import os
import numpy as np
from taming.loading_utils import load_config, load_model
from taming.data.custom import CustomTest as CustomDataset

from taming.models.LSFautoencoder import LSFVQVAE
from taming.plotting_utils import get_fig_pth

import torch

from omegaconf import OmegaConf
import argparse

from taming.analysis_utils import aggregate_metric_from_specimen_to_species, get_CosineDistance_matrix
from taming.plotting_utils import plot_heatmap


def save_cosine_distance_matrix(model, dataset, idx_to_label, plot_save_dir, device=None):
        encodings = []
        classes = []
        with torch.no_grad():
            for d in dataset:
                img = torch.tensor(d['image']).unsqueeze(0).permute(0, 3, 1, 2)
                if device is not None:
                    img = img.cuda()
                z, _, _ = model.image2encoding(img)
                z = (z @ model.LSF_disentangler.M.t())[:, :38]
                encodings.append(z)
                classes.append(d['class'])

        encodings = torch.cat(encodings)

        sort_indices = np.argsort(classes)
        sorted_encodings = encodings[sort_indices,:]
        sorted_classes = sorted(classes)

        z_cosine_distances = get_CosineDistance_matrix(sorted_encodings)

        os.makedirs(plot_save_dir, exist_ok=True)

        plot_heatmap(z_cosine_distances.cpu(), postfix=plot_save_dir,
                        title='z cosine distances')
        sorted_class_names_according_to_class_indx = [idx_to_label[x] for x in sorted_classes]
        embedding_dist = aggregate_metric_from_specimen_to_species(sorted_class_names_according_to_class_indx,
                                                                    z_cosine_distances)        
        plot_heatmap(embedding_dist.cpu(), postfix=plot_save_dir,
                        title='z cosine species distances')


@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE = configs_yaml.DEVICE
    size= configs_yaml.size
    file_list_path= configs_yaml.file_list_path    
    
    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    
    # Load model
    config = load_config(yaml_path, display=False)
    model = load_model(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=LSFVQVAE)

    idx_to_label = {y:x for x, y in dataset.labels_to_idx.items()}

    fig_path = get_fig_pth(ckpt_path)
    fig_path = os.path.join(str(fig_path), "heat_map_cosine_LSF")
    
    save_cosine_distance_matrix(model, dataset, idx_to_label, fig_path, device=DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/heatmapLSF.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)
