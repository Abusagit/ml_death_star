import torch
from torch_geometric.data import Dataset, Data
from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-r", "--root_dir")
    parser.add_argument("-p", "--file_postfix")
    parser.add_argument("-t", "--atom_distance_threshold", type=int)
    parser.add_argument("--y_prefix", action="store_true")
    
    
    return parser

class SpheresDataset(Dataset):
    def __init__(self, root, x_file_template, atom_proximity_max_radius=None, is_y_prefix=True, transform=None, pre_transform=None, pre_filter=None):
        """
        root = Where the dataset should be stored
        This folder is split into raw_dir (raw data) and processed_dir (processed data
        """
        
        self.atom_proximity_max_radius = atom_proximity_max_radius or np.inf
        self.x_file_template = "X_{}" + x_file_template + ".csv"
        
        self.y_template = x_file_template if is_y_prefix else ''
        
        super(SpheresDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        """
        If this file exists in raw_dir, the download is not triggered
        (download func is not implemented)
        """
        
        return f"Y_{self.y_template}.csv"
        
    def download(self):
        pass
    
    @property
    def processed_file_names(self):
        """If these files are found in [processed_directory], processing is stopped"""
        
        datafiles = list(map(lambda x: x.split("/")[-1], glob.glob(str(Path(self.processed_dir, "data_*")))))
#         datafiles = glob.glob("data_*")
#         breakpoint()
        return datafiles
        
    
    def process(self):
        raw_path = Path(self.raw_dir)
        processed_path = Path(self.processed_dir)
        
        labels = torch.tensor(np.loadtxt(raw_path / f"Y{self.y_template}.csv"))
        for label_index, label in tqdm(enumerate(labels), desc="Featurizing datapoints", total=labels.shape[0]):
            molecule_obj = np.loadtxt(raw_path / self.x_file_template.format(label_index+1), delimiter=',')
            
            node_features = self._get_node_features(molecule_obj)
            edge_index, edge_features = self._get_adjacency_edge_features(molecule_obj)
            
            data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        y=label)
            
            torch.save(data, processed_path / f"data_{label_index}.pt")
            
        print("_" * 100)
        return
    
    def _get_node_features(self, molecule_obj):
        """returns tensor with shape [num_nodes, num_features]"""
        
        
        molecule_obj[:, :3] = MinMaxScaler().fit_transform(molecule_obj[:, :3])
        charges = torch.tensor(molecule_obj, dtype=torch.float64).view(-1, 4)  # get shape [num_nodes, 1] as charhe is only one feature so far
        
        #charges = torch.tensor(molecule_obj).view(-1, 4)
        return charges
    def _get_adjacency_edge_features(self, molecule_obj):
        # process every edge between atoms and make undirectde edge representation\n
        edge_features = []
        
        sources = []
        destinations = []
        
        for i in range(molecule_obj.shape[0]):
            coords_i = molecule_obj[i, :3]
            
            for j in range(i + 1, molecule_obj.shape[0]):
                coords_j = molecule_obj[j, :3]
                
                dist = coords_i - coords_j
                
                distance_i_j = np.sqrt(dist @ dist)
                
                if distance_i_j > self.atom_proximity_max_radius:
                    continue
                    
                edge_features.append([distance_i_j])
                edge_features.append([distance_i_j])
                sources.append(i)
                sources.append(j)
                destinations.append(j)
                destinations.append(i)
                
        edge_features = torch.tensor(StandardScaler().fit_transform(edge_features), dtype=torch.float64).view(-1, 1)
        edge_index = torch.tensor(list(zip(sources, destinations)), dtype=torch.long).view(2, -1)
        return edge_index, edge_features
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(Path(self.processed_dir, f"data_{idx}.pt"))
        return data


if __name__ == "__main__":
    
    parser = get_parser()
    
    args = parser.parse_args()
    
    dataset = SpheresDataset(root=args.root_dir, x_file_template=args.file_postfix,
                             atom_proximity_max_radius=args.atom_distance_threshold,
                             is_y_prefix=args.y_prefix)
    