from sklearn.preprocessing import Normalizer, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
import pandas as pd

class DataProcessor:

    selected_features = ['source.location', 
                         'characteristics.tag.gender',
                         'characteristics.tag.tumor.size.maximumdiameter',
                         'characteristics.tag.stage.primary.tumor',
                         'characteristics.tag.stage.nodes', 
                         'characteristics.tag.stage.mets'] 

    def __init__(self):
        self.normalizer = Normalizer()
        self.scaler = StandardScaler()
        self.oh_encoder = OneHotEncoder()
        self.ordinal_encoder = OrdinalEncoder()
        # keep 95% of varaince
        self.pca = PCA(n_components = .95)

    def preprocess_gene_data(self, X):
        # format
        formatted = X.iloc[:,1:].T
        # normalize
        normalized = self.normalizer.fit_transform(formatted)
        # scale
        scaled = self.scaler.fit_transform(normalized)
        # reduce
        reduced = self.pca.fit_transform(scaled)
        reduced = pd.DataFrame(reduced)
        reduced.columns = [f"PCA_{i}" for i in reduced.columns]
        return reduced
    
    def preprocess_meta_data(self, meta):
        # separate numeric features
        numeric = "characteristics.tag.tumor.size.maximumdiameter"
        diameter = meta[numeric]
        meta = meta.drop(columns=numeric)
        # encode labels to integers
        encoded = self.ordinal_encoder.fit_transform(meta)
        # reconbine the dataframes
        encoded_meta = pd.DataFrame(encoded)
        encoded_meta.columns = meta.columns
        encoded_meta["diameter_tumor"] = diameter
        return encoded_meta
    
    def clean_metadata(self, meta):
        # subset selected features
        meta = meta[self.selected_features]
        # convert all to upper
        meta.iloc[:,-3:] = meta.iloc[:,-3:].apply(lambda col: col.str.upper())
        # fill nans with average
        avg = meta["characteristics.tag.tumor.size.maximumdiameter"].mean()
        meta.iloc[:,2] = meta.iloc[:,2].fillna(avg)        
        return meta
    
    def merge_gene_meta(self, X, meta):
        return meta.merge(X, how="inner", left_index=True, right_index=True)
    
    def fit_transform(self, X, metadata):
        meta = self.clean_metadata(metadata)
        preprocessed_meta = self.preprocess_meta_data(meta)
        preprocessed_genetic = self.preprocess_gene_data(X)
        merged_dataset = self.merge_gene_meta(preprocessed_genetic,
                                              preprocessed_meta)
        return merged_dataset