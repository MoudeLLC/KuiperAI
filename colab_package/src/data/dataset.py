"""
Dataset management and data loading utilities
"""
import numpy as np
from typing import List, Tuple, Optional, Iterator
import json
import os


class Dataset:
    """Base dataset class"""
    
    def __init__(self):
        self.data = []
        self.labels = []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx], self.labels[idx]
    
    def shuffle(self):
        """Shuffle the dataset"""
        indices = np.random.permutation(len(self))
        self.data = [self.data[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]


class TextDataset(Dataset):
    """Dataset for text data"""
    
    def __init__(self, texts: List[str], labels: Optional[List] = None,
                 vocab: Optional[dict] = None, max_length: int = 512):
        super().__init__()
        self.texts = texts
        self.max_length = max_length
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab
        
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        # Tokenize texts
        self.data = [self._tokenize(text) for text in texts]
        self.labels = labels if labels is not None else [0] * len(texts)
    
    def _build_vocab(self, texts: List[str]) -> dict:
        """Build vocabulary from texts"""
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for text in texts:
            for token in text.split():
                if token not in vocab:
                    vocab[token] = len(vocab)
        
        return vocab
    
    def _tokenize(self, text: str) -> np.ndarray:
        """Convert text to token indices"""
        tokens = text.split()[:self.max_length - 2]
        
        # Add special tokens
        token_ids = [self.vocab['<SOS>']]
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab['<UNK>']))
        token_ids.append(self.vocab['<EOS>'])
        
        # Pad to max_length
        while len(token_ids) < self.max_length:
            token_ids.append(self.vocab['<PAD>'])
        
        return np.array(token_ids, dtype=np.int32)
    
    def decode(self, token_ids: np.ndarray) -> str:
        """Convert token indices back to text"""
        tokens = []
        for idx in token_ids:
            if idx == self.vocab['<EOS>']:
                break
            if idx not in [self.vocab['<PAD>'], self.vocab['<SOS>']]:
                tokens.append(self.idx_to_token.get(idx, '<UNK>'))
        return ' '.join(tokens)
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    @classmethod
    def load_vocab(cls, filepath: str) -> dict:
        """Load vocabulary from file"""
        with open(filepath, 'r') as f:
            return json.load(f)


class DataLoader:
    """Data loader for batching and iteration"""
    
    def __init__(self, dataset: Dataset, batch_size: int = 32,
                 shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            self.dataset.shuffle()
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_end = min(i + self.batch_size, len(self.dataset))
            
            if self.drop_last and batch_end - i < self.batch_size:
                break
            
            batch_data = []
            batch_labels = []
            
            for j in range(i, batch_end):
                data, label = self.dataset[j]
                batch_data.append(data)
                batch_labels.append(label)
            
            # Stack into arrays
            batch_data = np.stack(batch_data)
            batch_labels = np.array(batch_labels)
            
            yield batch_data, batch_labels


class DataAugmenter:
    """Data augmentation utilities"""
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to data"""
        noise = np.random.randn(*data.shape) * noise_level
        return data + noise
    
    @staticmethod
    def random_crop(data: np.ndarray, crop_size: int) -> np.ndarray:
        """Randomly crop sequence"""
        if len(data) <= crop_size:
            return data
        
        start = np.random.randint(0, len(data) - crop_size)
        return data[start:start + crop_size]
    
    @staticmethod
    def token_dropout(tokens: np.ndarray, dropout_rate: float = 0.1,
                     pad_token: int = 0) -> np.ndarray:
        """Randomly drop tokens"""
        mask = np.random.rand(len(tokens)) > dropout_rate
        return np.where(mask, tokens, pad_token)


class KnowledgeBase:
    """Manages the knowledge base for training"""
    
    def __init__(self, base_dir: str = 'knowledge'):
        self.base_dir = base_dir
        self.datasets = {}
    
    def load_domain(self, domain: str) -> Dataset:
        """Load a specific knowledge domain"""
        domain_path = os.path.join(self.base_dir, 'datasets', domain)
        
        if not os.path.exists(domain_path):
            raise ValueError(f"Domain {domain} not found in knowledge base")
        
        # Load dataset files
        texts = []
        labels = []
        
        for filename in os.listdir(domain_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(domain_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(domain)
        
        dataset = TextDataset(texts, labels)
        self.datasets[domain] = dataset
        
        return dataset
    
    def get_all_domains(self) -> List[str]:
        """List all available domains"""
        datasets_path = os.path.join(self.base_dir, 'datasets')
        if not os.path.exists(datasets_path):
            return []
        
        return [d for d in os.listdir(datasets_path) 
                if os.path.isdir(os.path.join(datasets_path, d))]
    
    def create_mixed_dataset(self, domains: List[str]) -> Dataset:
        """Create a dataset mixing multiple domains"""
        all_texts = []
        all_labels = []
        
        for domain in domains:
            dataset = self.load_domain(domain)
            all_texts.extend(dataset.texts)
            all_labels.extend(dataset.labels)
        
        return TextDataset(all_texts, all_labels)
