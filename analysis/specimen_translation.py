
from scripts.analysis_utils import Embedding_Code_converter
from scripts.loading_utils import load_config, load_model
from scripts.data.custom import CustomTest as CustomDataset
import scripts.constants as CONSTANTS
from scripts.models.phyloautoencoder import PhyloVQVAE
from scripts.plotting_utils import get_fig_pth

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
from omegaconf import OmegaConf
import argparse
import tqdm
from pathlib import Path
from scipy.stats import entropy


# indexing of hist_arr: [code_location][raw list of codes of that location from all images]
# lowest entropy to highest entropy
def get_entropy_ordering(hist_arr_for_species):
    entropies = []
    for codes_forcode_location in hist_arr_for_species:
        value,counts = np.unique(codes_forcode_location, return_counts=True)
        entropies.append(entropy(counts))
    reverse_ordered_entropy_indices = np.argsort(entropies)
    # print(entropies, reverse_ordered_entropy_indices)
    return reverse_ordered_entropy_indices


class KeyImageEntropyHelper:
    def __init__(self, cb_per_level, n_phylolevels, n_nonphylocodes, get_code_reshaped_index):
        self.cb_per_level = cb_per_level
        self.n_phylolevels = n_phylolevels
        self.n_nonphylocodes = n_nonphylocodes
        self.get_code_reshaped_index = get_code_reshaped_index
        
        self.update = False
        self.changed_codes = []
        for i in range(self.n_phylolevels):
            self.changed_codes.append([])
    
    
    
    def new_update(self, indx, i):
        nonphylo_codes_count = self.cb_per_level*self.n_nonphylocodes

        if indx == nonphylo_codes_count-1:
            self.update = True
        elif indx>=nonphylo_codes_count:
            _, phylo_level = self.get_code_reshaped_index(i)
            self.changed_codes[phylo_level].append(i)
            if (phylo_level != self.n_phylolevels-1) and len(self.changed_codes[phylo_level]) == self.cb_per_level:
                self.update = True
                
        return self.update
        
    def isKeyImage(self, i):
        if self.update:
            self.update = False
            return True
        
        return False


@torch.no_grad()
def main(configs_yaml):
    yaml_path = configs_yaml.yaml_path
    ckpt_path = configs_yaml.ckpt_path
    DEVICE= configs_yaml.DEVICE
    size= configs_yaml.size
    file_list_path= configs_yaml.file_list_path
    file_list_path_interest = configs_yaml.file_list_path_interest
    show_only_key_imgs = configs_yaml.show_only_key_imgs

    # load image
    dataset = CustomDataset(size, file_list_path, add_labels=True)
    dataset_interest = CustomDataset(size, file_list_path_interest, add_labels=True)
    
    # Load model
    config = load_config(yaml_path, display=False)
    model = load_model(config, ckpt_path=ckpt_path, cuda=(DEVICE is not None), model_type=PhyloVQVAE)
    
    get_code_reshaped_index = model.phylo_disentangler.embedding_converter.get_code_reshaped_index
    n_phylocodes = model.phylo_disentangler.n_phylolevels*model.phylo_disentangler.codes_per_phylolevel

    for image_index1 in range(len(dataset_interest)):
        for image_index2 in range(len(dataset_interest)):
            if image_index1 >= image_index2:
                continue

            # img_indx_original1 = dataset.paths.index(dataset_interest.paths[image_index1])
            # img_indx_original2 = dataset.paths.index(dataset_interest.paths[image_index2])
            # print(img_indx_original1, image_index1, dataset_interest.paths[image_index1], dataset.paths[img_indx_original1])
            # print(img_indx_original2, image_index2, dataset_interest.paths[image_index2], dataset.paths[img_indx_original2])
            # raise


            classes = []
            codes = []
            dec_images = []
            filenames = []
            converter_phylo = None
            converter_nonattr = None
            len_phylo = None
            for img_indx in [image_index1, image_index2]:
                specimen = dataset_interest.data[img_indx]
                specimen_original = dataset.data[dataset.paths.index(dataset_interest.paths[img_indx])]
                processed_img = torch.Tensor(specimen['image']).unsqueeze(0).to(DEVICE)
                processed_img = processed_img.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                
                dec_image, _, _, in_out_disentangler = model(processed_img)
                dec_images.append(dec_image)
                if img_indx == image_index1:
                    dec_images.insert(0, processed_img)
                else:
                    dec_images.append(processed_img)
                    
                q_phylo = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_OUTPUT] 
                q_non_attr = in_out_disentangler[CONSTANTS.QUANTIZED_PHYLO_NONATTRIBUTE_OUTPUT] 

                if converter_phylo is None:
                    converter_phylo = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_phylo[0, :, :, :].shape)
                    converter_nonattr = Embedding_Code_converter(model.phylo_disentangler.quantize.get_codebook_entry_index, model.phylo_disentangler.quantize.embedding, q_non_attr[0, :, :, :].shape)
                
                all_code_indices_phylo = converter_phylo.get_phylo_codes(q_phylo[0, :, :, :].unsqueeze(0))
                all_code_indices_nonattr = converter_nonattr.get_phylo_codes(q_non_attr[0, :, :, :].unsqueeze(0))
                if len_phylo is None:
                    len_phylo = all_code_indices_phylo.view(1, -1).shape[-1]
                
                all_code = torch.cat([all_code_indices_phylo, all_code_indices_nonattr], dim=1)
                codes.append(all_code)
                
                species_index = specimen_original['class']
                filenames.append(os.path.basename(specimen['file_path_']))
                classes.append(species_index)
                print(specimen['file_path_'], specimen_original['class'])
                
                
            changes = []
            code1_modified =torch.clone(codes[0])
            code1 = codes[0]
            code2 = codes[1]

            # Get code and location ordering
            histograms_file = os.path.join(get_fig_pth(ckpt_path, postfix=CONSTANTS.HISTOGRAMS_FOLDER), CONSTANTS.HISTOGRAMS_FILE)
            histogram_file_exists = os.path.exists(histograms_file)
            if not histogram_file_exists:
                raise "histograms have not been generated. Run code_histogram.py first! Defaulting to index ordering"
            hist_arr, hist_arr_nonattr = pickle.load(open(histograms_file, "rb"))
            
            # by entrop over levels
            target_class = classes[-1]
            entropy_nonphylo = np.flip(get_entropy_ordering(hist_arr_nonattr[target_class]) + n_phylocodes)
            hist_phylo = hist_arr[target_class]
            hist_levels = []
            for i in range(n_phylocodes):
                hist_levels.append([])
            for i in range(len(hist_phylo)):
                relative_indx, lvl = get_code_reshaped_index(i)
                hist_levels[lvl].append(hist_phylo[i])
            for lvl in range(model.phylo_disentangler.n_phylolevels):
                hist_levels[lvl] = np.flip(get_entropy_ordering(hist_levels[lvl]))
            entropy_phylo = np.zeros(len(hist_phylo), dtype=int)
            o = 0
            for lvl in range(model.phylo_disentangler.n_phylolevels):
                for relative_index in range(model.phylo_disentangler.codes_per_phylolevel):
                    abs_indx = get_code_reshaped_index(hist_levels[lvl][relative_index], lvl)
                    entropy_phylo[o] = abs_indx
                    o = o+1
            order_ = np.concatenate((entropy_nonphylo, entropy_phylo))

            if show_only_key_imgs:
                key_image_helper = KeyImageEntropyHelper(model.phylo_disentangler.codes_per_phylolevel, model.phylo_disentangler.n_phylolevels, model.phylo_disentangler.n_levels_non_attribute, get_code_reshaped_index)
                
            for indx_, i in enumerate(order_):    
                if show_only_key_imgs:
                    key_image_helper.new_update(indx_, i)
                    
                if (not key_image_helper.update) and code1[0, i] == code2[0, i]:
                    continue
                
                code1_modified[0, i] = code2[0, i]
                
                z_phylo = converter_phylo.get_phylo_embeddings(code1_modified[:, :len_phylo])
                z_nonphylo = converter_nonattr.get_phylo_embeddings(code1_modified[:, len_phylo:])
                dec_image, _, _, in_out_disentangler = model.decode(z_phylo, z_nonphylo)
                
                
                if (not show_only_key_imgs) or (key_image_helper.isKeyImage(i)):
                    dec_images.insert(-2, dec_image)
                    
                    change = {
                        'code_index': i,
                        'phylo_level': 'level' + str(get_code_reshaped_index(i)[1]+1) if i < n_phylocodes else 'nonphylo',
                        'from_code': code1[0, i].detach().item(),
                        'to_code': code2[0, i].detach().item(),
                        'code_match_%': 100-torch.cdist(code1_modified.float(), code2.float(), p=0).item()*100/code2.shape[1]
                    }
                    
                    changes.append(change)
                
            fig = plt.figure(1, (2.5*len(dec_images), 4.))
            grid = ImageGrid(fig, 111,
                            nrows_ncols=(1, len(dec_images)),
                            )
            
            img_index = 0
            change_index = 0
            for axes in grid:  
                img_ = dec_images[img_index]
                img = ((np.fliplr(img_.squeeze().T.detach().cpu().numpy())+1.)*127.5).astype(np.uint8)
                axes.imshow(img, aspect='auto')
                if img_index==0:
                    title = "source image"
                elif img_index == len(dec_images)-1:
                    title = "target image"
                # else:#TODO: remove me:
                #     title = ""

                if img_index > 0 and img_index < len(dec_images)-1:
                    if img_index > 1 and img_index < len(dec_images)-2:
                        change = changes[change_index]
                        title = "replacing {}".format(change['phylo_level'])
                        change_index = change_index+ 1
                    else:
                        if img_index==1:
                            title="initial reconstruction"
                        else:
                            title="replacing species level"
                    
                axes.set_title(title, fontdict=None, loc='center', color = "k")
                axes.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
                    
                img_index = img_index + 1
                    
            plt.show()
                    
            fig_path = get_fig_pth(ckpt_path)
            filename = "{} to {}.png".format(image_index1, image_index2)
            fig_path = os.path.join(str(fig_path), "transitions", dataset.indx_to_label[classes[0]] + ' to ' + dataset.indx_to_label[classes[1]])
            Path(fig_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(os.path.join(fig_path, filename),bbox_inches='tight',dpi=300)
            
            plt.close()
                    
                    
                    
            
            
        

        
            
        
            
        
    
        
        
    
        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--config",
        type=str,
        nargs="?",
        const=True,
        default="analysis/configs/specimen_translation.yaml",
    )
    
    cfg, _ = parser.parse_known_args()
    configs = OmegaConf.load(cfg.config)
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(configs, cli)
    print(config)
    
    main(config)