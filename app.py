import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================================
# CONFIG
# =========================================

ANIME_PATH = "anime.csv"    # Kaggle anime metadata
RATING_PATH = "rating.csv"  # Kaggle user ratings

st.set_page_config(
    page_title="Anime Recommender - Hybrid + Neural CF",
    layout="wide",
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================
# DATA LOADING & PREPROCESSING
# =========================================

@st.cache_data
def load_data_and_models(
    min_ratings_per_user: int = 50,
    min_ratings_per_item: int = 100,
    max_users_for_cf: int = 2000,
):
    # Load raw data
    anime = pd.read_csv(ANIME_PATH)
    rating = pd.read_csv(RATING_PATH)

    # Remove "no rating" entries (-1)
    rating = rating[rating["rating"] != -1]

    # Filter by minimum activity
    user_counts = rating["user_id"].value_counts()
    item_counts = rating["anime_id"].value_counts()

    active_users = user_counts[user_counts >= min_ratings_per_user].index
    popular_items = item_counts[item_counts >= min_ratings_per_item].index

    rating_filtered = rating[
        rating["user_id"].isin(active_users)
        & rating["anime_id"].isin(popular_items)
    ].copy()

    # Optionally restrict to most active users (for speed of CF + NCF)
    user_counts2 = rating_filtered["user_id"].value_counts().sort_values(ascending=False)
    if len(user_counts2) > max_users_for_cf:
        keep_users = user_counts2.iloc[:max_users_for_cf].index
        rating_filtered = rating_filtered[rating_filtered["user_id"].isin(keep_users)]

    # Recompute items that survived
    item_counts2 = rating_filtered["anime_id"].value_counts()
    keep_items = item_counts2.index
    rating_filtered = rating_filtered[rating_filtered["anime_id"].isin(keep_items)]

    # Keep only anime present in filtered ratings
    anime_small = anime[anime["anime_id"].isin(keep_items)].copy()

    # Create user-item matrix (for CF + hybrid)
    user_item = rating_filtered.pivot_table(
        index="user_id", columns="anime_id", values="rating"
    )

    # ===== Item-based similarity (for CF & hybrid) =====
    user_item_filled = user_item.fillna(0)
    item_sim = cosine_similarity(user_item_filled.T)
    item_sim_df = pd.DataFrame(
        item_sim, index=user_item.columns, columns=user_item.columns
    )

    # ===== Content features (genres + type) =====
    anime_small["genre"] = anime_small["genre"].fillna("")
    anime_small["type"] = anime_small["type"].fillna("Unknown")

    genre_dummies = anime_small["genre"].str.get_dummies(sep=",")
    genre_dummies = genre_dummies.rename(columns=lambda x: x.strip())

    type_dummies = pd.get_dummies(anime_small["type"], prefix="type")

    item_features = pd.concat(
        [anime_small[["anime_id", "name"]].reset_index(drop=True),
         genre_dummies.reset_index(drop=True),
         type_dummies.reset_index(drop=True)],
        axis=1,
    )

    feature_cols = [c for c in item_features.columns if c not in ["anime_id", "name"]]

    # Align feature matrix with user_item columns (anime_id order)
    item_features = (
        item_features.set_index("anime_id")
        .loc[user_item.columns]
        .reset_index()
    )
    feature_cols = [c for c in item_features.columns if c not in ["anime_id", "name"]]
    item_feature_matrix = item_features.set_index("anime_id")[feature_cols]

    return (
        rating_filtered,
        anime_small,
        user_item,
        item_sim_df,
        item_features,
        feature_cols,
        item_feature_matrix,
    )


(
    rating_filtered,
    anime_small,
    user_item,
    item_sim_df,
    item_features,
    feature_cols,
    item_feature_matrix,
) = load_data_and_models()

# Lookup table for pretty printing
anime_lookup = anime_small.set_index("anime_id")[["name", "genre", "type"]]


# =========================================
# HYBRID RECOMMENDER (item-CF + contenu)
# =========================================

def get_item_based_cf_scores(user_id: int, k_neighbors: int = 20) -> pd.Series:
    """
    Simple item-based CF:
    For each candidate anime, aggregate ratings of similar animes
    already rated by the user.
    Returns a pandas Series: index = anime_id, value = predicted rating (≈1–10).
    """
    if user_id not in user_item.index:
        raise ValueError(f"user_id {user_id} not in model.")

    user_ratings_row = user_item.loc[user_id]
    rated_items = user_ratings_row[user_ratings_row.notna()].index
    candidate_items = user_item.columns.difference(rated_items)

    preds = {}

    for anime_id in candidate_items:
        sims = item_sim_df.loc[anime_id, rated_items]

        # take top-k similar items that user rated
        top_sims = sims.nlargest(min(k_neighbors, len(sims)))
        # filter out zero sims
        top_sims = top_sims[top_sims > 0]

        if top_sims.empty:
            continue

        top_ratings = user_ratings_row[top_sims.index]

        denom = np.abs(top_sims).sum()
        if denom == 0:
            continue

        pred = np.dot(top_sims.values, top_ratings.values) / denom
        preds[anime_id] = pred

    return pd.Series(preds)


def get_content_based_scores(
    user_id: int,
    like_threshold: float = 7.0,
) -> pd.Series:
    """
    Content-based on genres + type.
    Build a user profile from liked anime, then compute cosine similarity
    between profile and all anime feature vectors.
    Returns a Series: index = anime_id, value = similarity score.
    """
    if user_id not in user_item.index:
        raise ValueError(f"user_id {user_id} not in model.")

    user_ratings = rating_filtered[rating_filtered["user_id"] == user_id]

    if user_ratings.empty:
        return pd.Series(dtype=float)

    liked = user_ratings[user_ratings["rating"] >= like_threshold]
    if liked.empty:
        liked = user_ratings.sort_values("rating", ascending=False).head(10)

    # average rating per anime for this user
    user_item_ratings = liked.groupby("anime_id")["rating"].mean()

    # intersect with available feature items
    available_items = item_feature_matrix.index.intersection(user_item_ratings.index)
    if available_items.empty:
        return pd.Series(dtype=float)

    feat_mat = item_feature_matrix.loc[available_items].values
    weights = user_item_ratings.loc[available_items].values

    # weighted average profile
    profile = np.average(feat_mat, axis=0, weights=weights).reshape(1, -1)

    # cosine similarity to all items
    all_feats = item_feature_matrix.values
    sims = cosine_similarity(profile, all_feats)[0]

    return pd.Series(sims, index=item_feature_matrix.index)


def _normalize_to_0_1(series: pd.Series, index) -> pd.Series:
    """
    Normalize a series to [0,1] over the given index (candidates).
    """
    if series is None or len(index) == 0:
        return pd.Series(0.0, index=index)

    s = series.reindex(index)
    # if all NaN
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=index)
    s = s.fillna(s.min())
    min_v = s.min()
    max_v = s.max()
    if max_v == min_v:
        return pd.Series(0.5, index=index)  # all equal
    return (s - min_v) / (max_v - min_v)


def recommend_hybrid(
    user_id: int,
    top_n: int = 10,
    alpha: float = 0.5,
    k_neighbors: int = 20,
) -> pd.DataFrame:
    """
    Weighted hybrid:
    hybrid_score = alpha * normalized_CF + (1 - alpha) * normalized_content
    Then scaled to [0,5] for output.
    """
    cf_scores = get_item_based_cf_scores(user_id, k_neighbors=k_neighbors)
    cb_scores = get_content_based_scores(user_id)

    # Candidates = items not yet rated by user
    user_ratings_row = user_item.loc[user_id]
    rated_items = user_ratings_row[user_ratings_row.notna()].index
    candidates = item_feature_matrix.index.difference(rated_items)

    cf_norm = _normalize_to_0_1(cf_scores, candidates)
    cb_norm = _normalize_to_0_1(cb_scores, candidates)

    hybrid_0_1 = alpha * cf_norm + (1.0 - alpha) * cb_norm
    # scale to [0,5]
    hybrid_0_5 = hybrid_0_1 * 5.0

    top = hybrid_0_5.sort_values(ascending=False).head(top_n)

    result = anime_lookup.loc[top.index].copy()
    result["score_0_5"] = top.values

    return result.reset_index().rename(columns={"index": "anime_id"})


# =========================================
# AI-BASED APPROACH (CHAPTER 7: NCF)
# =========================================

class RatingsDataset(Dataset):
    def __init__(self, user_indices, item_indices, ratings_0_5):
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.ratings_0_5 = ratings_0_5

    def __len__(self):
        return len(self.ratings_0_5)

    def __getitem__(self, idx):
        return (
            self.user_indices[idx],
            self.item_indices[idx],
            self.ratings_0_5[idx],
        )


class NCF(nn.Module):
    """
    Neural Collaborative Filtering model:
    - User & item embeddings
    - MLP on concatenated embeddings
    - Sigmoid output, scaled to [0,5]
    """

    def __init__(self, n_users, n_items, embed_dim=32, hidden_dims=(64, 32)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        layers = []
        input_dim = 2 * embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.mlp = nn.Sequential(*layers)

        self.out = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_idx, item_idx):
        # user_idx, item_idx: LongTensor [batch]
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        x = torch.cat([u, i], dim=-1)
        h = self.mlp(x)
        logit = self.out(h)
        score_0_5 = self.sigmoid(logit) * 5.0  # [0,5]
        return score_0_5.squeeze(-1)


def build_ncf_training_data(ratings_df, max_users=None, max_items=None):
    """
    ratings_df: DataFrame with columns [user_id, anime_id, rating]
    Optionally limit to first max_users / max_items (for faster training).
    """
    df = ratings_df.copy()

    if max_users is not None:
        top_users = df["user_id"].value_counts().head(max_users).index
        df = df[df["user_id"].isin(top_users)]

    if max_items is not None:
        top_items = df["anime_id"].value_counts().head(max_items).index
        df = df[df["anime_id"].isin(top_items)]

    unique_users = df["user_id"].unique()
    unique_items = df["anime_id"].unique()

    user2idx = {u: idx for idx, u in enumerate(unique_users)}
    item2idx = {i: idx for idx, i in enumerate(unique_items)}

    user_idx = df["user_id"].map(user2idx).values.astype("int64")
    item_idx = df["anime_id"].map(item2idx).values.astype("int64")

    # Target ratings in [0,5]
    ratings_0_5 = (df["rating"].values.astype("float32") / 2.0).clip(0.0, 5.0)

    dataset = RatingsDataset(
        torch.from_numpy(user_idx),
        torch.from_numpy(item_idx),
        torch.from_numpy(ratings_0_5),
    )

    return dataset, user2idx, item2idx


def train_ncf_model(
    ratings_df,
    n_epochs=3,
    batch_size=2048,
    embed_dim=32,
    hidden_dims=(64, 32),
    lr=1e-3,
    max_users=2000,
    max_items=3000,
):
    dataset, user2idx, item2idx = build_ncf_training_data(
        ratings_df,
        max_users=max_users,
        max_items=max_items,
    )

    n_users = len(user2idx)
    n_items = len(item2idx)

    model = NCF(n_users, n_items, embed_dim=embed_dim, hidden_dims=hidden_dims).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for u, i, r in loader:
            u = u.to(device)
            i = i.to(device)
            r = r.to(device)

            optimizer.zero_grad()
            preds = model(u, i)
            loss = loss_fn(preds, r)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(r)

        epoch_loss = running_loss / len(dataset)
        print(f"[NCF] Epoch {epoch+1}/{n_epochs} - MSE loss: {epoch_loss:.4f}")

    return model, user2idx, item2idx


@st.cache_resource(show_spinner="Training AI-based model (Neural CF / Chap. 7)...")
def get_trained_ncf_model(ratings_df):
    model, user2idx, item2idx = train_ncf_model(
        ratings_df,
        n_epochs=3,
        batch_size=2048,
        embed_dim=32,
        hidden_dims=(64, 32),
        lr=1e-3,
        max_users=2000,
        max_items=3000,
    )
    return model, user2idx, item2idx


ncf_model, ncf_user2idx, ncf_item2idx = get_trained_ncf_model(rating_filtered)


def recommend_ai_ncf(
    user_id: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    AI-based approach (Chap. 7):
    Neural Collaborative Filtering with embeddings + MLP.
    Output scores are already in [0,5].
    """
    if user_id not in user_item.index:
        raise ValueError(f"user_id {user_id} not found in CF data.")

    if user_id not in ncf_user2idx:
        # User not seen during training → return empty result
        return pd.DataFrame(columns=["anime_id", "name", "genre", "type", "score_0_5"])

    user_idx = ncf_user2idx[user_id]

    user_ratings_row = user_item.loc[user_id]
    rated_items = user_ratings_row[user_ratings_row.notna()].index

    # Candidates = items not yet rated by the user AND known to the NCF model
    all_items = user_item.columns
    candidates = [aid for aid in all_items if (aid not in rated_items and aid in ncf_item2idx)]

    if not candidates:
        return pd.DataFrame(columns=["anime_id", "name", "genre", "type", "score_0_5"])

    user_tensor = torch.tensor(
        [user_idx] * len(candidates),
        dtype=torch.long,
        device=device,
    )
    item_tensor = torch.tensor(
        [ncf_item2idx[aid] for aid in candidates],
        dtype=torch.long,
        device=device,
    )

    ncf_model.eval()
    with torch.no_grad():
        scores = ncf_model(user_tensor, item_tensor).cpu().numpy()  # already [0,5]

    score_series = pd.Series(scores, index=candidates)
    top = score_series.sort_values(ascending=False).head(top_n)

    result = anime_lookup.loc[top.index].copy()
    result["score_0_5"] = top.values

    return result.reset_index().rename(columns={"index": "anime_id"})


# =========================================
# STREAMLIT UI
# =========================================

# Restrict user selection to users seen by BOTH CF and NCF
available_user_ids = sorted(set(user_item.index).intersection(ncf_user2idx.keys()))

st.title("🎌 Anime Recommendation System")

st.sidebar.header("⚙️ Navigation & Settings")

page = st.sidebar.radio(
    "Page",
    ("Overview", "Recommendations", "Comparison"),
)

selected_user = st.sidebar.selectbox(
    "Choose a user ID",
    options=available_user_ids,
)

# Common control for how many recs
n_recs = st.sidebar.slider(
    "Number of recommendations (for Rec & Comparison)",
    min_value=5,
    max_value=30,
    value=10,
    step=1,
)

# --------- PAGE 1: OVERVIEW ---------
if page == "Overview":
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of users", len(user_item.index))
    with col2:
        st.metric("Number of anime", len(user_item.columns))
    with col3:
        st.metric("Number of ratings", len(rating_filtered))

    st.markdown("### Ratings distribution")
    ratings_hist = rating_filtered["rating"].value_counts().sort_index()
    st.bar_chart(ratings_hist)

    st.markdown("### Top 10 genres (by number of anime)")
    genre_series = (
        anime_small["genre"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
    )
    genre_counts = genre_series.value_counts().head(10)
    st.bar_chart(genre_counts)

    st.markdown("### Average rating by type")
    avg_by_type = (
        rating_filtered.merge(
            anime_small[["anime_id", "type"]],
            on="anime_id",
            how="left",
        )
        .groupby("type")["rating"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(avg_by_type)

    st.markdown("---")
    st.markdown(
        """
This overview page gives a quick visual summary of the anime dataset:
- Ratings distribution  
- Most frequent genres  
- Average rating per type (TV, Movie, OVA, ...)

Use the **Recommendations** page to see personalised lists, and the
**Comparison** page to compare the hybrid vs AI-based (Neural CF, Chap. 7) models.
"""
    )

# --------- PAGE 2: RECOMMENDATIONS ---------
elif page == "Recommendations":
    algo = st.sidebar.radio(
        "Recommendation approach",
        (
            "Hybrid (item-CF + content, weighted)",
            "AI-based (Neural Collaborative Filtering)",
        ),
    )

    if algo.startswith("Hybrid"):
        alpha = st.sidebar.slider(
            "Weight α for Collaborative Filtering (1 = only CF, 0 = only content)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        k_neighbors = st.sidebar.slider(
            "Number of neighbor anime (k) for item-based CF",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
        )

        st.subheader(
            f"Hybrid recommendations for user {selected_user} (scores in [0,5])"
        )
        with st.spinner("Computing hybrid recommendations..."):
            recs = recommend_hybrid(
                user_id=selected_user,
                top_n=n_recs,
                alpha=alpha,
                k_neighbors=k_neighbors,
            )

        st.write("Top recommended anime (Hybrid):")
        st.dataframe(
            recs[["anime_id", "name", "genre", "type", "score_0_5"]]
            .reset_index(drop=True)
        )

    else:
        st.subheader(
            f"AI-based recommendations (Neural CF, Chap. 7) for user {selected_user} (scores in [0,5])"
        )
        with st.spinner("Computing Neural CF recommendations..."):
            recs = recommend_ai_ncf(
                user_id=selected_user,
                top_n=n_recs,
            )

        if recs.empty:
            st.info("No AI-based recommendations available for this user.")
        else:
            st.write("Top recommended anime (AI-based Neural CF):")
            st.dataframe(
                recs[["anime_id", "name", "genre", "type", "score_0_5"]]
                .reset_index(drop=True)
            )

    st.markdown("---")
    st.subheader(f"📖 Rating history for user {selected_user} (sample)")
    user_hist = rating_filtered[rating_filtered["user_id"] == selected_user].copy()
    user_hist = user_hist.merge(
        anime_small[["anime_id", "name", "genre", "type"]],
        on="anime_id",
        how="left",
    )
    user_hist = user_hist.sort_values("rating", ascending=False).head(30)
    st.dataframe(
        user_hist[["anime_id", "name", "genre", "type", "rating"]]
        .reset_index(drop=True)
    )

# --------- PAGE 3: COMPARISON ---------
elif page == "Comparison":
    st.subheader(
        f"Comparison: Hybrid vs AI-based (Neural CF, Chap. 7) for user {selected_user}"
    )

    # Settings specific to hybrid for the comparison
    alpha_cmp = st.sidebar.slider(
        "α for Hybrid on Comparison page (1 = only CF, 0 = only content)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    k_neighbors_cmp = st.sidebar.slider(
        "k neighbors for item-based CF (Comparison)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )

    with st.spinner("Computing both recommendation lists..."):
        hybrid_recs = recommend_hybrid(
            user_id=selected_user,
            top_n=n_recs,
            alpha=alpha_cmp,
            k_neighbors=k_neighbors_cmp,
        )
        ai_recs = recommend_ai_ncf(
            user_id=selected_user,
            top_n=n_recs,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Hybrid recommendations")
        st.dataframe(
            hybrid_recs[["anime_id", "name", "genre", "type", "score_0_5"]]
            .rename(columns={"score_0_5": "hybrid_score"})
            .reset_index(drop=True)
        )

    with col2:
        st.markdown("### AI-based recommendations (Neural CF)")
        if ai_recs.empty:
            st.info("No AI-based recommendations available for this user.")
        else:
            st.dataframe(
                ai_recs[["anime_id", "name", "genre", "type", "score_0_5"]]
                .rename(columns={"score_0_5": "ai_score"})
                .reset_index(drop=True)
            )

    # Compare overlapping recommendations (same anime recommended by both)
    st.markdown("---")
    st.markdown("### Overlap between Hybrid and AI-based recommendations")

    hybrid_cmp = hybrid_recs[["anime_id", "score_0_5"]].rename(
        columns={"score_0_5": "hybrid_score"}
    )
    ai_cmp = ai_recs[["anime_id", "score_0_5"]].rename(
        columns={"score_0_5": "ai_score"}
    )

    overlap = hybrid_cmp.merge(ai_cmp, on="anime_id", how="inner")
    if not overlap.empty:
        overlap = overlap.merge(
            anime_small[["anime_id", "name"]],
            on="anime_id",
            how="left",
        )
        overlap = overlap[["anime_id", "name", "hybrid_score", "ai_score"]]
        st.write(
            "Anime that both systems recommend (with their scores scaled in [0,5]):"
        )
        st.dataframe(overlap.reset_index(drop=True))
    else:
        st.write(
            "For this user and these parameters, there is no overlap between the top-N lists."
        )

    st.markdown(
        """
The comparison page lets you:
- See the two lists side by side  
- Inspect which anime appear in both lists and how the scores differ  

This directly illustrates the difference between a **hybrid (item-CF + content)** system
and an **AI-based (Neural Collaborative Filtering, Chapter 7)** model.
"""
    )