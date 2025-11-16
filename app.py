import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

# --------- CONFIG ---------
ANIME_PATH = "anime.csv"
RATING_PATH = "rating.csv"

st.set_page_config(
    page_title="Anime Recommender - Hybrid + AI",
    layout="wide",
)

# --------- DATA LOADING & PREPROCESSING ---------
@st.cache_data
def load_data_and_models(
    min_ratings_per_user: int = 50,
    min_ratings_per_item: int = 100,
    max_users_for_model: int = 2000,
    nmf_components: int = 15,
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

    # Recompute after first filter
    user_counts2 = rating_filtered["user_id"].value_counts().sort_values(ascending=False)
    # Limit number of users for performance
    if len(user_counts2) > max_users_for_model:
        keep_users = user_counts2.iloc[:max_users_for_model].index
        rating_filtered = rating_filtered[rating_filtered["user_id"].isin(keep_users)]

    # Recompute items that survived
    item_counts2 = rating_filtered["anime_id"].value_counts()
    keep_items = item_counts2.index
    rating_filtered = rating_filtered[rating_filtered["anime_id"].isin(keep_items)]

    # Keep only anime present in filtered ratings
    anime_small = anime[anime["anime_id"].isin(keep_items)].copy()

    # Create user-item matrix
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

    # feature columns
    feature_cols = [c for c in item_features.columns if c not in ["anime_id", "name"]]

    # Align feature matrix with user_item columns (anime_id order)
    item_features = (
        item_features.set_index("anime_id")
        .loc[user_item.columns]
        .reset_index()
    )
    feature_cols = [c for c in item_features.columns if c not in ["anime_id", "name"]]
    item_feature_matrix = item_features.set_index("anime_id")[feature_cols]

    # ===== AI-based model: NMF latent factor model =====
    R = user_item.fillna(0).values
    nmf_model = NMF(
        n_components=nmf_components,
        init="random",
        random_state=42,
        max_iter=300,
    )
    user_factors = nmf_model.fit_transform(R)
    item_factors = nmf_model.components_
    R_hat = np.dot(user_factors, item_factors)

    nmf_pred = pd.DataFrame(R_hat, index=user_item.index, columns=user_item.columns)

    return (
        rating_filtered,
        anime_small,
        user_item,
        item_sim_df,
        item_features,
        feature_cols,
        item_feature_matrix,
        nmf_pred,
    )


(
    rating_filtered,
    anime_small,
    user_item,
    item_sim_df,
    item_features,
    feature_cols,
    item_feature_matrix,
    nmf_pred,
) = load_data_and_models()

available_user_ids = sorted(user_item.index.tolist())

# Small helper lookup (for pretty tables)
anime_lookup = anime_small.set_index("anime_id")[["name", "genre", "type"]]

# --------- RECOMMENDER IMPLEMENTATIONS ---------
def get_item_based_cf_scores(user_id: int, k_neighbors: int = 20) -> pd.Series:

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
    like_threshold: float = 5.0,
) -> pd.Series:

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


def recommend_nmf_ai(
    user_id: int,
    top_n: int = 10,
) -> pd.DataFrame:

    if user_id not in nmf_pred.index:
        raise ValueError(f"user_id {user_id} not in NMF model.")

    preds = nmf_pred.loc[user_id]

    user_ratings_row = user_item.loc[user_id]
    rated_items = user_ratings_row[user_ratings_row.notna()].index

    # Remove seen items
    preds = preds.drop(rated_items, errors="ignore")

    # Rescale to [0,5]
    preds_0_5 = np.clip(preds / 2.0, 0.0, 5.0)

    top = preds_0_5.sort_values(ascending=False).head(top_n)

    result = anime_lookup.loc[top.index].copy()
    result["score_0_5"] = top.values

    return result.reset_index().rename(columns={"index": "anime_id"})


# --------- STREAMLIT UI LAYOUT ---------
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

# --------- PAGE 1: OVERVIEW (Landing page with mini visualisation) ---------
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
    # Split and explode genres
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
- Average rating per anime type (TV, Movie, OVA, ...)

Use the **Recommendations** page to see personalized lists, and the
**Comparison** page to compare the hybrid vs AI-based approach.
"""
    )

# --------- PAGE 2: RECOMMENDATIONS ---------
elif page == "Recommendations":
    algo = st.sidebar.radio(
        "Recommendation approach",
        (
            "Hybrid (item-CF + content, weighted)",
            "AI-based (latent factor model - NMF)",
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

        st.write("Top recommended anime:")
        st.dataframe(
            recs[["anime_id", "name", "genre", "type", "score_0_5"]]
            .reset_index(drop=True)
        )

    else:
        st.subheader(
            f"AI-based recommendations (latent factors / NMF) for user {selected_user} (scores in [0,5])"
        )
        with st.spinner("Computing NMF-based recommendations..."):
            recs = recommend_nmf_ai(
                user_id=selected_user,
                top_n=n_recs,
            )

        st.write("Top recommended anime:")
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
        f"Comparison: Hybrid vs AI-based (scores in [0,5]) for user {selected_user}"
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
        ai_recs = recommend_nmf_ai(
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
        st.markdown("### AI-based recommendations (NMF)")
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
On this page you can:
- See the **two lists** side by side.
- Inspect which anime appears in **both top-N lists**, and how their scores differ.
This is a nice way to discuss the behaviour of a **hybrid** vs an **AI-based (latent factor)**
approach in your report / presentation.
"""
    )
