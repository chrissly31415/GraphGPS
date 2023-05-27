import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List, Optional

from rdkit import Chem
import numpy as np
from pathlib import Path
import pandas as pd

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)



class AQSOL(InMemoryDataset):
    r"""The AQSOL dataset from the `Benchmarking Graph Neural Networks
    <http://arxiv.org/abs/2003.00982>`_ paper based on
    `AqSolDB <https://www.nature.com/articles/s41597-019-0151-1>`_, a
    standardized database of 9,982 molecular graphs with their aqueous
    solubility values, collected from 9 different data sources.

    The aqueous solubility targets are collected from experimental measurements
    and standardized to LogS units in AqSolDB. These final values denote the
    property to regress in the :class:`AQSOL` dataset. After filtering out few
    graphs with no bonds/edges, the total number of molecular graphs is 9,833.
    For each molecular graph, the node features are the types of heavy atoms
    and the edge features are the types of bonds between them, similar as in
    the :class:`~torch_geometric.datasets.ZINC` dataset.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in
            the final dataset. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - 9,833
          - ~17.6
          - ~35.8
          - 1
          - 1
    """
    url = 'https://www.dropbox.com/s/lzu9lmukwov12kt/aqsol_graph_raw.zip?dl=1'

    def __init__(self, root: str, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'atom_dict.pickle',
            'bond_dict.pickle'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'asqol_graph_raw'), self.raw_dir)
        os.unlink(path)

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, 'rb') as f:
                graphs = pickle.load(f)

            data_list: List[Data] = []
            for graph in graphs:
                x, edge_attr, edge_index, y = graph

                x = torch.from_numpy(x).view(-1, 1)
                edge_attr = torch.from_numpy(edge_attr)
                edge_index = torch.from_numpy(edge_index)
                y = torch.tensor([y]).float()

                if edge_index.numel() == 0:
                    continue  # Skipping for graphs with no bonds/edges.
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list), path)

    def atoms(self) -> List[str]:
        return [
            'Br', 'C', 'N', 'O', 'Cl', 'Zn', 'F', 'P', 'S', 'Na', 'Al', 'Si',
            'Mo', 'Ca', 'W', 'Pb', 'B', 'V', 'Co', 'Mg', 'Bi', 'Fe', 'Ba', 'K',
            'Ti', 'Sn', 'Cd', 'I', 'Re', 'Sr', 'H', 'Cu', 'Ni', 'Lu', 'Pr',
            'Te', 'Ce', 'Nd', 'Gd', 'Zr', 'Mn', 'As', 'Hg', 'Sb', 'Cr', 'Se',
            'La', 'Dy', 'Y', 'Pd', 'Ag', 'In', 'Li', 'Rh', 'Nb', 'Hf', 'Cs',
            'Ru', 'Au', 'Sm', 'Ta', 'Pt', 'Ir', 'Be', 'Ge'
        ]

    def bonds(self) -> List[str]:
        return ['NONE', 'SINGLE', 'DOUBLE', 'AROMATIC', 'TRIPLE']



import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class ZINC(InMemoryDataset):
    r"""The ZINC dataset from the `ZINC database
    <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_ and the
    `"Automatic Chemical Design Using a Data-Driven Continuous Representation
    of Molecules" <https://arxiv.org/abs/1610.02415>`_ paper, containing about
    250,000 molecular graphs with up to 38 heavy atoms.
    The task is to regress the penalized :obj:`logP` (also called constrained
    solubility in some works), given by :obj:`y = logP - SAS - cycles`, where
    :obj:`logP` is the water-octanol partition coefficient, :obj:`SAS` is the
    synthetic accessibility score, and :obj:`cycles` denotes the number of
    cycles with more than six atoms.
    Penalized :obj:`logP` is a score commonly used for training molecular
    generation models, see, *e.g.*, the
    `"Junction Tree Variational Autoencoder for Molecular Graph Generation"
    <https://proceedings.mlr.press/v80/jin18a.html>`_ and
    `"Grammar Variational Autoencoder"
    <https://proceedings.mlr.press/v70/kusner17a.html>`_ papers.

    Args:
        root (str): Root directory where the dataset should be saved.
        subset (bool, optional): If set to :obj:`True`, will only load a
            subset of the dataset (12,000 molecular graphs), following the
            `"Benchmarking Graph Neural Networks"
            <https://arxiv.org/abs/2003.00982>`_ paper. (default: :obj:`False`)
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - ZINC Full
          - 249,456
          - ~23.2
          - ~49.8
          - 1
          - 1
        * - ZINC Subset
          - 12,000
          - ~23.2
          - ~49.8
          - 1
          - 1
    """

    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(
        self,
        root: str,
        subset: bool = True,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self) -> str:
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                mol = mols[idx]
                x = mol['atom_type'].to(torch.long).view(-1, 1)
                y = mol['logP_SA_cycle_normalized'].to(torch.float)
                adj = mol['bond_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))


class OPERA(InMemoryDataset):
    
    url = 'https://ndownloader.figstatic.com/files/10692997'
       
    # Define a mapping from bond types to integers
    bond_type_to_int = {Chem.rdchem.BondType.SINGLE: 1, 
                    Chem.rdchem.BondType.DOUBLE: 2,
                    Chem.rdchem.BondType.TRIPLE: 3,
                    Chem.rdchem.BondType.AROMATIC: 1}
    
    
    DATASETS = ["OPERA_LogP","OPERA_AOH", "OPERA_BCF", "OPERA_BioHL", "OPERA_BP", "OPERA_HL", "OPERA_KM", "OPERA_KOA", "OPERA_KOC", "OPERA_MP", "OPERA_RBioDeg", "OPERA_VP", "OPERA_WS"]

    TARGET_MAP = {'VP': 'LogVP', 'WS': 'LogMolar','KOC':'LogKOC'}

    PREDEFINED_MAPPING = {
    6: 0,  # Carbon
    1: 1,  # Hydrogen
    7: 2,  # Nitrogen
    8: 3,  # Oxygen
    9: 4,  # Fluor
    14: 5, # Si
    15: 6, # P
    16: 7, # Sulfur
    17: 8, # Chlorine
    34: 9, # Selen
    35: 10,# Bromine
    50: 11, # Sn
    }
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        name: str = 'OPERA_LogP',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        assert split in ['train', 'test']
        self.name = name
        assert self.name in self.DATASETS
        self.target = self.name.replace("OPERA_","")
        if self.target in self.TARGET_MAP.keys():
            self.target = self.TARGET_MAP[self.target]
        super().__init__(root, transform, pre_transform, pre_filter) # Dataset starts download and process on init
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        raw_test = [file.name for file in Path(self.raw_dir).glob('*') if file.name.startswith("TR_") and file.name.endswith(".sdf")]
        raw_train = [file.name for file in Path(self.raw_dir).glob('*') if file.name.startswith("TST_") and file.name.endswith(".sdf")]
        return raw_train + raw_test
            

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed', self.name)

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'test.pt']


    @classmethod
    def encode_atom_types(cls,mol: Chem.rdchem.Mol) -> List[int]:
        mol_atom_types = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            mol_atom_types.append(cls.PREDEFINED_MAPPING.get(atomic_num, len(cls.PREDEFINED_MAPPING)))

        return mol_atom_types

    def sdf2molecules(self,sd_file_path: str) -> List[dict]:
        supplier = Chem.SDMolSupplier(sd_file_path)
        mols = []
        for mol in supplier:
            if mol is not None:
                if self.target:
                    y = np.asarray(float(mol.GetProp(self.target)))
                else:
                    y = [1.0]
                mol_atom_types = np.asarray(OPERA.encode_atom_types(mol),dtype=np.int8) # encoding important!
                bond_adj_matrix = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.int8)
                    # Fill in the adjacency matrix with bond types
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bond_type = self.bond_type_to_int[bond.GetBondType()]
                    bond_adj_matrix[i, j] = bond_type
                    bond_adj_matrix[j, i] = bond_type 
                
                num_atom = len(mol_atom_types)
                if num_atom>40: continue
                mols.append({'num_atom':num_atom,'atom_type': mol_atom_types,'bond_type': bond_adj_matrix, 'y': y})
        return mols
    

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, self.name), self.raw_dir)
        os.unlink(path)


    def process(self):     
        for i,split in enumerate(['train', 'test']):
            
            sd_file_path = osp.join(self.raw_dir, self.raw_file_names[i])
            mols = self.sdf2molecules(sd_file_path)
            print(f"Molecule count: {len(mols)}")
            pbar = tqdm(total=len(mols))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for mol in mols:

                x = torch.from_numpy(mol['atom_type']).to(torch.long).view(-1, 1) # if not casted here this gives 
                y = torch.from_numpy(mol['y']).to(torch.float)
                
                adj = torch.from_numpy(mol['bond_type']).to(torch.long)
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))


class SDF(InMemoryDataset):
    
    # Define a mapping from bond types to integers
    bond_type_to_int = {Chem.rdchem.BondType.SINGLE: 1, 
                    Chem.rdchem.BondType.DOUBLE: 2,
                    Chem.rdchem.BondType.TRIPLE: 3,
                    Chem.rdchem.BondType.AROMATIC: 1}
    
    PREDEFINED_MAPPING = {
    6: 0,  # Carbon
    1: 1,  # Hydrogen
    7: 2,  # Nitrogen
    8: 3,  # Oxygen
    9: 4,  # Fluor
    14: 5, # Si
    15: 6, # P
    16: 7, # Sulfur
    17: 8, # Chlorine
    34: 9, # Selen
    35: 10,# Bromine
    50: 11, # Sn
    }
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        name: str = '/NEWDATA/moldata/sampl6/sampl6.sdf',
        target: Optional[str] = "logP", 
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        assert split in ['train', 'test','val']
        self.name = name
        self.target = target
        self.is_csv = self.name.endswith(".csv")
      
        super().__init__(root, transform, pre_transform, pre_filter) # Dataset starts download and process on init
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.name]
            

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'test.pt']


    def encode_atom_types(self,mol: Chem.rdchem.Mol) -> List[int]:
        mol_atom_types = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            mol_atom_types.append(self.PREDEFINED_MAPPING.get(atomic_num, len(self.PREDEFINED_MAPPING)))

        return mol_atom_types


    def encode_mol(self,mol):
        mol_atom_types = np.asarray(self.encode_atom_types(mol),dtype=np.int8) # encoding important!
        bond_adj_matrix = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.int8)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = self.bond_type_to_int[bond.GetBondType()]
            bond_adj_matrix[i, j] = bond_type
            bond_adj_matrix[j, i] = bond_type 
                
        num_atom = len(mol_atom_types)
        return mol_atom_types, bond_adj_matrix, num_atom
     
     
    def sdf2molecules(self,sd_file_path: str, max_atoms: int = 40) -> List[dict]:
        supplier = Chem.SDMolSupplier(sd_file_path)
        mols = []
        for mol in supplier:
            if mol is not None:
                if self.target:
                    y = np.asarray(float(mol.GetProp(self.target)))
                else:
                    y = np.asarray([1.0])
                mol_atom_types, bond_adj_matrix, num_atom = self.encode_mol(mol)             
                if num_atom>max_atoms: continue
                mols.append({'num_atom':num_atom,'atom_type': mol_atom_types,'bond_type': bond_adj_matrix, 'y': y})
        return mols
      
      
    def csv2molecules(self,csv_path: str,  max_atoms: int = 40) ->List[dict]:
        df = pd.read_csv(csv_path,sep=";") # infer separator
        print(df)
        if not self.target in df.columns:
            print(f"Missing target in columns: {self.target}")
            breakpoint()
        
        if "SMILES" in df.columns:
            df['rdmol'] = df['SMILES'].map(lambda x: Chem.MolFromSmiles(x))

        mols = []
        for i,row in df.iterrows():
            mol = row['rdmol']
            if self.target in row:   
                y = np.asarray(row[self.target])
            else:
                y = np.asarray([1.0])
            mol_atom_types, bond_adj_matrix, num_atom = self.encode_mol(mol)  
            if num_atom>max_atoms: continue
            mols.append({'num_atom':num_atom,'atom_type': mol_atom_types,'bond_type': bond_adj_matrix, 'y': y})
        return mols

    def download(self):
        shutil.rmtree(self.raw_dir)
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)

        shutil.copy(self.name,self.raw_dir)


    def process(self):     
        for i,split in enumerate(['train']):
            file_path = osp.join(self.raw_dir, self.raw_file_names[i])
            if self.name.endswith(".sdf"): 
                mols = self.sdf2molecules(file_path)
            elif self.name.endswith(".csv"):
                mols = self.csv2molecules(file_path)
            else:
                print("Need either csv or sdf file!")
                return
                
            print(f"Molecule count: {len(mols)}")
            pbar = tqdm(total=len(mols))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for mol in mols:

                x = torch.from_numpy(mol['atom_type']).to(torch.long).view(-1, 1) # if not casted here this gives 
                y = torch.from_numpy(mol['y']).to(torch.float)
                
                adj = torch.from_numpy(mol['bond_type']).to(torch.long)
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))


if __name__ == '__main__':
    #dataset = OPERA("./datasets/OPERA",name="OPERA_LogP")
    dataset = SDF("./datasets/SAMPL6",name = "/NEWDATA/moldata/sampl6/sampl6.sdf")
    #dataset = ZINCMOD("./datasets/ZINCMOD")
    #dataset.download()