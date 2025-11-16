# 🎌 Anime Recommender – Hybrid & AI-based

A small anime recommendation system built with Python and Streamlit using:

* a **hybrid approach**: item–item collaborative filtering + content-based filtering (genres + type)
* an **AI-based approach**: Neural Collaborative Filtering (**NCF**) model

Both approaches output scores normalised to **[0,5]**.

---


## 📦 Dataset

Kaggle: **Anime Recommendations Database**
[https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

Place the following files at the project root (not committed to git):

* `anime.csv`
* `rating.csv`

---

## 🛠 Installation & Run

Using **uv**:

```bash
uv init
uv add streamlit pandas numpy scikit-learn
uv run streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🧱 Features

* **Overview**: basic stats and visualisations of the dataset
* **Recommendations**: personalised recommendations (Hybrid or NCF)
* **Comparison**: side-by-side comparison of the two approaches and their scores

---

Dataset under Kaggle licence, used for academic purposes only.
