import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

UNWEIGHTED_TABLE = {
    "same_author": 1.0, "same_location": 1.0, "same_topic": 1.0,
    "same_language": 1.0, "same_media_type": 1.0, "same_toxicity_bin": 1.0,
}

RELATION_COLUMN_MAP = {
    "same_author":       "user_id",
    "same_location":     "location",
    "same_topic":        "topic",
    "same_language":     "language",
    "same_media_type":   "media_type",
    "same_toxicity_bin": "toxic_bin",
}


def compute_relation_weights(train_attrs_df, label_col="popularity", w_min=0.5, w_max=2.0, seed=42):
    """
    Algorithm 1, step 1-2: data-driven, leakage-safe weight derivation.
    CHỈ nhận attribute thô của TRAIN split (1 seed, 1 platform) -- không
    hề nhìn vào test set hay vào feature importance của model nào khác.
    """
    df = train_attrs_df.copy()
    if "toxicity_score" in df.columns:
        df["toxic_bin"] = (df["toxicity_score"] / 10).round() * 10

    mi_scores = {}
    for relation, col in RELATION_COLUMN_MAP.items():
        if col not in df.columns:
            continue
        sub = df[[col, label_col]].dropna()
        if sub[col].nunique() < 2:
            mi_scores[relation] = 0.0
            continue
        x_enc = LabelEncoder().fit_transform(sub[col].astype(str)).reshape(-1, 1)
        mi = mutual_info_classif(x_enc, sub[label_col].values,
                                  discrete_features=True, random_state=seed)[0]
        mi_scores[relation] = mi

    vals = np.array(list(mi_scores.values()))
    lo, hi = vals.min(), vals.max()
    weights = {}
    for relation, mi in mi_scores.items():
        weights[relation] = (w_min + w_max) / 2 if hi - lo < 1e-12 else \
            w_min + (mi - lo) / (hi - lo) * (w_max - w_min)
    return weights, mi_scores


def build_homophily_edges(nodes_df, train_ids, weight_table, sort_key="followers"):
    df = nodes_df.copy()
    if "toxicity_score" in df.columns:
        df["toxic_bin"] = (df["toxicity_score"] / 10).round() * 10

    df_sorted = df.sort_values(by=sort_key, ascending=True)
    train_ids = set(train_ids)

    edges = []
    for relation, col in RELATION_COLUMN_MAP.items():
        if col not in df_sorted.columns or relation not in weight_table:
            continue
        w = weight_table[relation]
        for val, group in df_sorted.groupby(col):
            if pd.isna(val):
                continue
            ids = group["post_id"].values
            if len(ids) < 2:
                continue
            for i in range(len(ids) - 1):
                a, b = ids[i], ids[i + 1]
                if (a not in train_ids) and (b not in train_ids):
                    continue
                edges.append({"Source": a, "Target": b, "Weight": w, "Relation": relation})

    edges_df = pd.DataFrame(edges, columns=["Source", "Target", "Weight", "Relation"])
    if not edges_df.empty:
        st = np.sort(edges_df[["Source", "Target"]].values, axis=1)
        edges_df["Source"], edges_df["Target"] = st[:, 0], st[:, 1]
        edges_df = edges_df.drop_duplicates(subset=["Source", "Target", "Relation"])
        edges_df = edges_df[edges_df["Source"] != edges_df["Target"]]
    return edges_df.reset_index(drop=True)


def build_similarity_knn_edges(nodes_df, train_ids, numeric_cols, k=5):
    df = nodes_df.copy().reset_index(drop=True)
    train_mask = df["post_id"].isin(set(train_ids)).values

    scaler = StandardScaler()
    train_feats = scaler.fit_transform(df.loc[train_mask, numeric_cols].fillna(0))
    all_feats = scaler.transform(df[numeric_cols].fillna(0))

    n_train = int(train_mask.sum())
    k_eff = min(k + 1, n_train)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(train_feats)
    _, indices = nn.kneighbors(all_feats)

    train_post_ids = df.loc[train_mask, "post_id"].values
    all_post_ids = df["post_id"].values

    edges = []
    for i, node_id in enumerate(all_post_ids):
        for rank in range(k_eff):
            neighbor_id = train_post_ids[indices[i, rank]]
            if neighbor_id == node_id:
                continue
            edges.append({"Source": node_id, "Target": neighbor_id, "Weight": 1.0, "Relation": "knn_similarity"})

    edges_df = pd.DataFrame(edges, columns=["Source", "Target", "Weight", "Relation"])
    if not edges_df.empty:
        st = np.sort(edges_df[["Source", "Target"]].values, axis=1)
        edges_df["Source"], edges_df["Target"] = st[:, 0], st[:, 1]
        edges_df = edges_df.drop_duplicates(subset=["Source", "Target"])
        edges_df = edges_df[edges_df["Source"] != edges_df["Target"]]
    return edges_df.reset_index(drop=True)


def compute_centrality_features(edges_df, all_node_ids, betweenness_k=300, seed=42):
    G = nx.Graph()
    G.add_nodes_from(all_node_ids)
    for _, row in edges_df.iterrows():
        G.add_edge(row["Source"], row["Target"], weight=row["Weight"])

    k = betweenness_k if (betweenness_k and betweenness_k < G.number_of_nodes()) else None

    degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True, k=k, seed=seed)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, weight="weight")
    try:
        eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes()}

    communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    community_map = {}
    for cid, com in enumerate(communities):
        for n in com:
            community_map[n] = cid

    feat_df = pd.DataFrame({
        "post_id": list(G.nodes()),
        "degree_centrality": [degree[n] for n in G.nodes()],
        "betweenness_centrality": [betweenness[n] for n in G.nodes()],
        "closeness_centrality": [closeness[n] for n in G.nodes()],
        "pagerank": [pagerank[n] for n in G.nodes()],
        "eigenvector_centrality": [eigenvector[n] for n in G.nodes()],
        "modularity_class": [community_map.get(n, -1) for n in G.nodes()],
    })
    return feat_df, G


def compute_node2vec_embeddings(G, dimensions=64, walk_length=20, num_walks=20, workers=1, seed=42):
    from node2vec import Node2Vec
    n2v = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks,
                    workers=workers, seed=seed, quiet=True)
    model = n2v.fit(window=5, min_count=1, batch_words=4)
    node_ids = list(model.wv.index_to_key)
    embeddings = model.wv.vectors
    emb_df = pd.DataFrame(embeddings, columns=[f"n2v_{i}" for i in range(dimensions)])
    emb_df.insert(0, "post_id", node_ids)
    return emb_df


def evaluate_predictions(y_test, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def shap_summary(model, X_sample, max_display=15, save_path=None):
    import shap
    import matplotlib.pyplot as plt
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return shap_values


def visualize_subgraph_sample(G, train_ids, sample_seed_node=None, n_hops=2, max_nodes=60, save_path=None, seed=42):
    """
    Lấy 1 cụm nhỏ quanh 1 node để kiểm tra trực quan: cạnh có nối đúng group
    không, và (quan trọng nhất) node test (màu cam) có bao giờ nối trực tiếp
    với node test khác không -- nếu có, đó là dấu hiệu leakage còn sót.
    """
    rng = np.random.default_rng(seed)
    if sample_seed_node is None:
        candidates = [n for n in G.nodes() if G.degree(n) > 0]
        sample_seed_node = rng.choice(candidates)

    nodes_to_keep = {sample_seed_node}
    frontier = {sample_seed_node}
    for _ in range(n_hops):
        next_frontier = set()
        for nd in frontier:
            next_frontier.update(G.neighbors(nd))
        nodes_to_keep.update(next_frontier)
        frontier = next_frontier
        if len(nodes_to_keep) >= max_nodes:
            break

    nodes_to_keep = list(nodes_to_keep)[:max_nodes]
    sub_G = G.subgraph(nodes_to_keep)

    train_ids = set(train_ids)
    node_colors = ["#4C9AFF" if nd in train_ids else "#FF8A65" for nd in sub_G.nodes()]
    edge_widths = [sub_G[u][v]["weight"] for u, v in sub_G.edges()]

    # tự động cảnh báo nếu có cạnh test-test trong subsample này
    test_test_edges = [(u, v) for u, v in sub_G.edges() if u not in train_ids and v not in train_ids]
    if test_test_edges:
        print(f"  CẢNH BÁO: phát hiện {len(test_test_edges)} cạnh test-test: {test_test_edges}")
    else:
        print("  OK: không có cạnh test-test trong subsample này.")

    pos = nx.spring_layout(sub_G, seed=seed)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(sub_G, pos, node_color=node_colors, node_size=300)
    nx.draw_networkx_edges(sub_G, pos, width=edge_widths)
    nx.draw_networkx_labels(sub_G, pos, font_size=6)
    plt.title(f"Subgraph around {sample_seed_node}  (xanh=train, cam=test)")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    return sub_G




def train_eval_variant(train_df, test_df, drop_cols=("post_id", "user_id", "popularity"), seed=42):
    drop_cols = [c for c in drop_cols if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols).fillna(0)
    y_train = train_df["popularity"]
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0).fillna(0)
    y_test = test_df["popularity"]

    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    ratio = neg / pos

    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=ratio, n_estimators=500, learning_rate=0.02,
        max_depth=6, min_child_weight=1, gamma=0.1,
        subsample=0.8, colsample_bytree=0.5, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_predictions(y_test, y_pred, y_prob), model, X_test