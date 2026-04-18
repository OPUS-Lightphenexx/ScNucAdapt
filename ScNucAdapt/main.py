import random

import torch
from matplotlib import pyplot as plt, cm
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

seed =34
import numpy as np
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
#np.set_printoptions(threshold=np.inf)


from ScUniverse.Utils import CS, split_and_merge_op, Arg, pairwise_distance
import numpy as np
array = np.load(r'')
y = np.load(r'',allow_pickle=True)
array = torch.tensor(array,dtype=torch.float32)
print(np.unique(y))
array2 = np.load(r'')
y2 =np.load(r'',allow_pickle=True)
print(np.unique(y2))
array2 = torch.tensor(array2,dtype=torch.float32)

y_label = np.concatenate([y,y2])
unique_y_label = np.unique(y_label)
label_to_num = {lab: idx for idx,lab in enumerate(unique_y_label)}

y = [label_to_num[l] for l in y]
y2 = [label_to_num[l] for l in y2]
print(np.unique(y2))

y2 = torch.LongTensor(y2)
y = torch.LongTensor(y)
print(np.bincount(y))
print(np.bincount(y2))

hyperparameter=400000000000000
# ---------------------------
# 2) Simple Autoencoder
# ---------------------------
class Base(nn.Module):
    def __init__(self, input_dim, latent_dim=50):  # 2D latent for direct plotting
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),


            nn.Linear(32, latent_dim)
        )

        self.classifier =nn.Sequential(
            nn.Linear(latent_dim, 3)
        )
    def forward(self, x):
        z = self.encoder(x)
        classify = self.classifier(z)

        return z,classify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Base(input_dim=array.shape[1], latent_dim=40).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

counts = torch.tensor(np.bincount(y))
print(counts)
weights = 1.0 / counts
weights = weights / weights.sum() * len(counts)
print(weights)
weights = torch.tensor([0.0537, 2.3022, 6.6441])
classify_loss = nn.CrossEntropyLoss(weight=weights.to(device))


X_tensor = array.to(device)
array = array.to(device)
y = y.to(device)
array2 = array2.to(device)
y2 = y2.to(device)
source_dataset = TensorDataset(X_tensor, y)
source_dataloader = DataLoader(source_dataset, batch_size=1028)

# ---------------------------
# 3) Train Autoencoder (no batches)
# ---------------------------
for epoch in range(52000):  # small number for demo
    loss_calculate = 0

    if epoch >=200:

        if epoch %1 == 0:
            with torch.no_grad():
                x_target, logit_target = model(array2)
                z,logit = model(array)

                z_draw = z.cpu().detach().numpy()
                x_target_draw = x_target.cpu().detach().numpy()
                draw = np.concatenate([z_draw, x_target_draw], axis=0)
                label_draw = np.concatenate([y.cpu().detach().numpy(), y2.cpu().detach().numpy()])
                import matplotlib.pyplot as plt
                import umap
                import numpy as np

                reducer = umap.UMAP(
                    n_neighbors=10,
                    metric='cosine',
                    min_dist=0.5,
                    n_components=2,
                    random_state=42,
                )
                draw = reducer.fit_transform(draw)
                batch_label_new = np.array([0] * len(array) + [1] * len(array2))
                id_to_cell = {v: k for k, v in label_to_num.items()}
                label_draw = [id_to_cell[i] for i in label_draw]
                label_draw = np.array(label_draw)
                all_celltypes = sorted(label_to_num.keys())
                cmap = cm.get_cmap('tab20', len(all_celltypes))
                celltype_to_color = {
                    cell: cmap(i)
                    for i, cell in enumerate(all_celltypes)
                }
                batch_dict = {'ScRNA_seq': 0, 'SnRNA_seq': 1}
                batch_to_cell = {v: k for k, v in batch_dict.items()}
                batch_label = [batch_to_cell[i] for i in batch_label_new]
                batch_label = np.array(batch_label)
                batch_colors = {
                    'ScRNA_seq': '#1f77b4',  # blue
                    'SnRNA_seq': '#ff7f0e'  # orange
                }
                label_draw_unique = set(label_draw)
                batch_draw_unique = set(batch_label)

                fig, ax = plt.subplots(figsize=(8, 5))
                for cell in sorted(set(label_draw)):  # sorted = stable order
                    idx = label_draw == cell  # boolean mask (fast)
                    ax.scatter(
                        draw[idx, 0],
                        draw[idx, 1],
                        label=cell,
                        s=3,
                        color=celltype_to_color[cell]
                    )
                #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=5)
                plt.tight_layout()
                plt.show()
                fig, ax = plt.subplots(figsize=(8, 5))
                for batch in sorted(set(batch_label)):
                    idx = batch_label == batch
                    ax.scatter(
                        draw[idx, 0],
                        draw[idx, 1],
                        label=batch,
                        s=3,
                        color=batch_colors[batch]
                    )
                #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=10)
                plt.tight_layout()
                plt.show()

                centroid = split_and_merge_op(x_target, Arg(init_K=4)).to('cuda')
                print('done')

                dist = pairwise_distance(x_target, centroid)
                value, pred = torch.min(dist, dim=1)
                target_dataset = TensorDataset(array2, pred)
                target_dataloader = DataLoader(target_dataset, batch_size=1028)

            # import matplotlib.pyplot as plt

            with torch.no_grad():
                merge_decision = []
                known_cell_type = []
                z_source, logit_source = model(array)
                z_target, logit_target= model(array2)

                for i in range(len(torch.unique(pred))):
                    distance_save = []
                    target_data = z_target[pred == i]
                    for j in range(len(torch.unique(y))):
                        source_data = z_source[y == j]
                        divergence = CS(source_data, target_data)
                        distance_save.append([i, j, divergence])
                        known_cell_type.append(j)

                    min_third = float('inf')  # Set to infinity initially
                    min_array = None

                    for k in distance_save:
                        if k[2].cpu().detach().numpy() < min_third:
                            min_third = k[2].cpu().detach().numpy()
                            min_array = k

                    merge_decision.append(min_array)
                print(merge_decision)

        with torch.no_grad():
            x_target_known, logit_target_known = model(array2)
            know_cells_pred = torch.argmax(logit_target_known, dim=1)
            acc = accuracy_score(y_true=y2.cpu().detach().numpy(), y_pred=know_cells_pred.cpu().detach().numpy())
            print(acc)
            f1 = f1_score(y_true=y2.cpu().detach().numpy(), y_pred=know_cells_pred.cpu().detach().numpy(),average='macro')
            print(f1)
            confusion = confusion_matrix(y_true=y2.cpu().detach().numpy(), y_pred=know_cells_pred.cpu().detach().numpy())
            print(confusion)

        for batch_idx, ((data_s,label_s),(data_t,label_t_pred)) in enumerate(zip(source_dataloader, target_dataloader)):
            optimizer.zero_grad()
            z_source_batch, logit_source_batch = model(data_s)
            z_target_batch, logit_target_batch = model(data_t)
            loss = classify_loss(logit_source_batch, label_s)

            cs_loss = 0.0

            for p in merge_decision:
                cs_loss = cs_loss + CS(z_source_batch[label_s == p[1]], z_target_batch[label_t_pred == p[0]])
            if not torch.isnan(cs_loss):
                loss = loss + 0.4* cs_loss
            loss_calculate += loss
            loss.backward()
            optimizer.step()
    else:
        for batch_idx, (data_s,label_s) in enumerate(source_dataloader):
            optimizer.zero_grad()
            z_source_batch,logit_source_batch = model(data_s)
            loss = classify_loss(logit_source_batch, label_s)
            loss_calculate += loss
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            x_target_known, logit_target_known = model(array2)
            know_cells_pred = torch.argmax(logit_target_known, dim=1)
            acc = accuracy_score(y_true=y2.cpu().detach().numpy(), y_pred=know_cells_pred.cpu().detach().numpy())
            print(acc)
            f1 = f1_score(y_true=y2.cpu().detach().numpy(), y_pred=know_cells_pred.cpu().detach().numpy(),average='macro')
            print(f1)

    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss_calculate/len(source_dataloader):.4f}")
