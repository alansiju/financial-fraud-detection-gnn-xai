import os
import json, sys, torch, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing_pipeline as pp_mod

# ---- Preprocessor inspection ----
print()
print("=" * 60)
print("PREPROCESSOR")
print("=" * 60)

import pickle
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'FraudPreprocessor':
            return pp_mod.FraudPreprocessor
        return super().find_class(module, name)

with open('outputs/models/preprocessor.pkl', 'rb') as f:
    pp = CustomUnpickler(f).load()

print(f"n_features    : {pp.n_features}")
print(f"num_cols count: {len(pp.num_cols)}")
print(f"cat_cols count: {len(pp.cat_cols)}")
print(f"all_cols count: {len(pp.all_cols)}")
print(f"fitted        : {pp.fitted}")
print(f"num_cols[:10] : {pp.num_cols[:10]}")
print(f"cat_cols[:10] : {pp.cat_cols[:10]}")
print(f"all_cols[:5]  : {pp.all_cols[:5]}")
# verify scaler only fitted on numeric
print(f"Scaler fitted on: {pp.scaler.n_features_in_} features (should == {len(pp.num_cols)})")
print(f"ScalerMean[0:3]  : {pp.scaler.mean_[:3]}")
print(f"Encoder n_cats   : {[len(c) for c in pp.encoder.categories_[:5]]}")

# ---- Graph inspection ----
print()
print("=" * 60)
print("GRAPH (fraud_graph.pt)")
print("=" * 60)
data = torch.load('fraud_graph.pt', map_location='cpu', weights_only=False)
N = data.x.shape[0]
E = data.edge_index.shape[1]
print(f"Nodes                 : {N:,}")
print(f"Edges (stored, 2xE)   : {E:,}")
print(f"Edges (undirected)    : {E//2:,}")
print(f"Node features         : {data.x.shape[1]}")
print(f"x dtype               : {data.x.dtype}")
print(f"y dtype               : {data.y.dtype}")
print(f"Fraud labels total    : {data.y.sum().item():,} ({100*data.y.float().mean().item():.2f}%)")
print(f"train_mask size       : {data.train_mask.sum().item():,}")
print(f"val_mask size         : {data.val_mask.sum().item():,}")
print(f"test_mask size        : {data.test_mask.sum().item():,}")
train_fraud = data.y[data.train_mask].float().mean().item()
val_fraud   = data.y[data.val_mask].float().mean().item()
test_fraud  = data.y[data.test_mask].float().mean().item()
print(f"Fraud rate (train)    : {100*train_fraud:.2f}%")
print(f"Fraud rate (val)      : {100*val_fraud:.2f}%")
print(f"Fraud rate (test)     : {100*test_fraud:.2f}%")
print(f"Mask overlap check    : {(data.train_mask & data.val_mask).sum().item()} (should=0)")
print(f"Mask overlap check    : {(data.train_mask & data.test_mask).sum().item()} (should=0)")
print(f"Mask overlap check    : {(data.val_mask & data.test_mask).sum().item()} (should=0)")
print(f"x contains NaN        : {torch.isnan(data.x).any().item()}")
print(f"x contains Inf        : {torch.isinf(data.x).any().item()}")
# check feature consistency
print(f"x.shape[1] == pp.n_features: {data.x.shape[1]} == {pp.n_features} -> {data.x.shape[1] == pp.n_features}")
