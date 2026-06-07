import os
import pandas as pd
import requests

class DatasetManager:
    """
    Manages loading, downloading, caching, and preprocessing of PROMISE Software Engineering datasets.
    Supported datasets: Ant, Camel, JEdit, Lucene, Tomcat
    """
    
    DATASETS = {
        "ant": {
            "name": "Apache Ant 1.7",
            "url": "https://raw.githubusercontent.com/feiwww/PROMISE-backup/master/bug-data/ant/ant-1.7.csv",
            "filename": "ant-1.7.csv"
        },
        "camel": {
            "name": "Apache Camel 1.6",
            "url": "https://raw.githubusercontent.com/feiwww/PROMISE-backup/master/bug-data/camel/camel-1.6.csv",
            "filename": "camel-1.6.csv"
        },
        "jedit": {
            "name": "jEdit 4.3",
            "url": "https://raw.githubusercontent.com/feiwww/PROMISE-backup/master/bug-data/jedit/jedit-4.3.csv",
            "filename": "jedit-4.3.csv"
        },
        "lucene": {
            "name": "Apache Lucene 2.4",
            "url": "https://raw.githubusercontent.com/feiwww/PROMISE-backup/master/bug-data/lucene/lucene-2.4.csv",
            "filename": "lucene-2.4.csv"
        },
        "tomcat": {
            "name": "Apache Tomcat 6.0",
            "url": "https://raw.githubusercontent.com/klainfo/DefectData/master/inst/extdata/terapromise/ck/tomcat.csv",
            "filename": "tomcat-6.0.csv"
        }
    }
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default to data folder inside workspace d:\code
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_dataset(self, key: str) -> str:
        """
        Downloads a dataset from raw github source and caches it locally.
        """
        if key not in self.DATASETS:
            raise ValueError(f"Unknown dataset key: {key}. Supported keys are: {list(self.DATASETS.keys())}")
            
        info = self.DATASETS[key]
        filepath = os.path.join(self.data_dir, info["filename"])
        
        if not os.path.exists(filepath):
            print(f"Downloading {info['name']} from {info['url']}...")
            response = requests.get(info["url"])
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Saved to {filepath}")
        return filepath

    def load_dataset(self, key: str) -> pd.DataFrame:
        """
        Loads a dataset as a Pandas DataFrame, downloading it if not cached.
        """
        filepath = self.download_dataset(key)
        df = pd.read_csv(filepath)
        return df

    def preprocess_dataset(self, df: pd.DataFrame):
        """
        Preprocesses a PROMISE dataset.
        Splits columns into:
        - Metadata (name, version)
        - Features (WMC, DIT, NOC, CBO, RFC, LCOM, LOC, etc.)
        - Target ('bug' and binary 'is_buggy')
        """
        # Create a copy to avoid mutating original
        df = df.copy()
        
        # Strip whitespace from column names if any
        df.columns = [col.strip() for col in df.columns]
        
        # PROMISE datasets typically have:
        # 'name', 'version', metrics..., 'bug'
        metadata_cols = []
        for col in ['name', 'version', 'class']:
            if col in df.columns:
                metadata_cols.append(col)
                
        # Find bug column (usually 'bug')
        target_col = None
        for col in ['bug', 'defects', 'defect']:
            if col in df.columns:
                target_col = col
                break
                
        if target_col is None:
            # Fallback to last column if 'bug' not found
            target_col = df.columns[-1]
            
        # Target: binary classification (1 = buggy, 0 = clean)
        # In PROMISE, bug column contains count of defects (or yes/no strings).
        bug_values = df[target_col]
        if bug_values.dtype == object:
            df['is_buggy'] = bug_values.astype(str).str.lower().isin(['yes', 'true', '1', 'buggy']).astype(int)
        else:
            df['is_buggy'] = (pd.to_numeric(bug_values, errors='coerce').fillna(0) > 0).astype(int)
        
        # Feature columns: numerical columns that are not metadata or target
        exclude_cols = metadata_cols + [target_col, 'is_buggy']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Convert feature columns to numeric, filling NaN with 0
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        return df, feature_cols, 'is_buggy', metadata_cols
