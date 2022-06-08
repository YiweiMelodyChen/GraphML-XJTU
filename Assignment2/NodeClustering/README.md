# README

## Dataset

[`KarateClub`](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.KarateClub)

 Zachary's karate club network from the ["An Information Flow Model for Conflict and Fission in Small Groups"](http://www1.ind.ku.dk/complexLearning/zachary1977.pdf) paper, containing 34 nodes, connected by 156 (undirected and unweighted) edges.

## Result

| MODEL   | Calinski-Harabaz Index | Silhouette Coefficient |
| ------- | ---------------------- | ---------------------- |
| GCN     | 34.85395874702531      | 0.4443395              |
| **GAT** | **38.46212906145406**  | **0.4816**             |
| SAGE    | 14.700669094454609     | 0.35009575             |
| RGCN    | 37.447999736749445     | 0.45621094             |

