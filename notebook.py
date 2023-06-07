# %%
import torch
import scanpy as sc
import numpy as np
from dataset import SingleCellRNACountsDataset
from dataprep import prep_sparse_data_for_training as prep_data_for_training
from encoder import EncodeZ, CompositeEncoder, EncodeNonZLatents
from decoder import Decoder
from train import run_training
from model import RemoveBackgroundPyroModel
import pyro
from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO
import consts as consts

# %%
from pathlib import Path
current_dir = Path(__file__).parent

adata = sc.read_10x_h5(current_dir / ".." / "example_data" / "MS466" / "filtered_feature_bc_matrix.h5")

# %%
model_type = "full"

# %%
# here we also compute all the priors
dataset_obj = SingleCellRNACountsDataset(
    input_file=str(current_dir / ".." / "example_data" / "MS466" / "filtered_feature_bc_matrix.h5"),
    model_name=model_type,
    exclude_antibodies=True,
    low_count_threshold=15,
    fpr=0.01,
    expected_cell_count=None,
    total_droplet_barcodes=25_000,
    fraction_empties=0.5
)
dataset_obj

# %%
dataset_obj.priors

# %%
dataset_obj.priors['chi_ambient']

# %%
count_matrix = dataset_obj.get_count_matrix()
# Configure pyro options (skip validations to improve speed).
pyro.enable_validation(False)
pyro.distributions.enable_validation(False)

# %%
# Load the dataset into DataLoaders.
frac = 1  # Fraction of barcodes to use for training
batch_size = int(min(300, frac * dataset_obj.analyzed_barcode_inds.size / 2))
batch_size

# %%
use_cuda = True
z_hidden_dims = [100]
d_hidden_dims = [10, 2]
p_hidden_dims = [100, 10]
z_dim = 10

# %%
fraction_empties = 0.5

# %%
learning_rate = 1e-4
epochs = 150

# %%
# Set up encode
encoder_z = EncodeZ(input_dim=count_matrix.shape[1],
                    hidden_dims=z_hidden_dims,
                    output_dim=z_dim,
                    input_transform='normalize')

encoder_other = EncodeNonZLatents(n_genes=count_matrix.shape[1],
                                    z_dim=z_dim,
                                    hidden_dims=consts.ENC_HIDDEN_DIMS,
                                    log_count_crossover=dataset_obj.priors['log_counts_crossover'],
                                    prior_log_cell_counts=np.log1p(dataset_obj.priors['cell_counts']),
                                    input_transform='normalize')

encoder = CompositeEncoder({'z': encoder_z,
                            'other': encoder_other})

# %%
test_X = torch.from_numpy(np.asarray(count_matrix.todense().astype(np.float32))).to(device='cuda')

# %%
encoder_z.forward(test_X)

# %%
# setu p decoder
decoder = Decoder(input_dim=z_dim,
                    hidden_dims=z_hidden_dims[::-1],
                    output_dim=count_matrix.shape[1])

# %%
# set up the model
model = RemoveBackgroundPyroModel(model_type="full",
                                  encoder=encoder,
                                  decoder=decoder,
                                  dataset_obj=dataset_obj,
                                  use_cuda=use_cuda)
model

# %%
# set up train and test loader
train_loader, test_loader = \
    prep_data_for_training(dataset=count_matrix,
                            empty_drop_dataset=dataset_obj.get_count_matrix_empties(),
                            random_state=dataset_obj.random,
                            batch_size=batch_size,
                            training_fraction=frac,
                            fraction_empties=fraction_empties,
                            shuffle=True,
                            use_cuda=use_cuda)

# %%
# Set up the optimizer.
optimizer = pyro.optim.clipped_adam.ClippedAdam
optimizer_args = {'lr': learning_rate, 'clip_norm': 10.}
optimizer

# %%
# Set up a learning rate scheduler.
minibatches_per_epoch = int(np.ceil(len(train_loader) / train_loader.batch_size).item())
scheduler_args = {'optimizer': optimizer,
                    'max_lr': learning_rate * 10,
                    'steps_per_epoch': minibatches_per_epoch,
                    'epochs': epochs,
                    'optim_args': optimizer_args}
scheduler = pyro.optim.OneCycleLR(scheduler_args)

# %%
model.model_type

# %%
if model.model_type == "simple":
    loss_function = Trace_ELBO()
else:
    loss_function = TraceEnum_ELBO(max_plate_nesting=1)
loss_function

# %%
svi = SVI(model.model, model.guide, scheduler, loss=loss_function)
svi

# %%
train_elbo, test_elbo, succeeded = run_training(model, svi, train_loader, test_loader,
                                                epochs=epochs, test_freq=5,
                                                final_elbo_fail_fraction=None,
                                                epoch_elbo_fail_fraction=None)

# %%


# %%


# %%



