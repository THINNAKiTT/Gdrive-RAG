"""
Usage:
    # Export the whole collection (metadata + text preview) to CSV:
    ```uv run python chroma_viewer.py --export-csv output.csv```

    # Launch an interactive web viewer (table + embedding graph),
    # on a separate port from the main app (default 8502, so both can run at the same time):
    ```uv run python chroma_viewer.py --web```
    ```uv run python chroma_viewer.py --web --port 8600```
"""
import argparse
import csv
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _load_collection_rows(db_path: str):
    from src.storage.vector_db import VectorDBManager

    db_manager = VectorDBManager(db_path=db_path)
    data = db_manager.chroma_collection.get(
        include=["metadatas", "documents", "embeddings"]
    )
    return data.get("metadatas", []), data.get("documents", []), data.get("ids", []), data.get("embeddings", [])

def export_csv(output_path: str, db_path: str):
    print(f"Reading collection from {db_path}...")
    metadatas, documents, ids, _ = _load_collection_rows(db_path)

    if not metadatas:
        print("Collection is empty -- nothing to export.")
        return 

    fieldnames = [
        "chroma_id",
        "file_id",
        "file_name",
        "page_number",
        "modified_time",
        "source",
        "web_view_link",
        "text_preview",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chroma_id, meta, doc in zip(ids, metadatas, documents):
            writer.writerow({
                "chroma_id": chroma_id,
                "file_id": meta.get("file_id", ""),
                "file_name": meta.get("file_name", ""),
                "page_number": meta.get("page_number", ""),
                "modified_time": meta.get("modified_time", ""),
                "source": meta.get("source", ""),
                "web_view_link": meta.get("web_view_link", ""),
                "text_preview": (doc or "")[:300].replace("\n", " ")
            })

    print(f"Exported {len(metadatas)} chunk(s) to {output_path}")

def launch_web(port: int):
    this_file = os.path.abspath(__file__)
    cmd = [
        "streamlit", "run", this_file,
        "--server.port", str(port),
        "--",
        "--render-web-ui",
    ]
    print(f"Launching viewer at http://localhost:{port}")
    subprocess.run(cmd)

def _render_web_ui():
    import streamlit as st
    import pandas as pd

    st.set_page_config(page_title="ChromaDB Viewer", layout="wide")
    st.title("ChromaDB Viewer")

    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    with st.spinner("Loading collection..."):
        metadatas, documents, ids, embeddings = _load_collection_rows(db_path)

        if not metadatas:
            st.warning(f"No data found in collection at `{db_path}`.")
            st.stop()

        df = pd.DataFrame({
            "chroma_id": ids,
            "file_id": [m.get("file_id", "") for m in metadatas],
            "file_name": [m.get("file_name", "") for m in metadatas],
            "page_number": [m.get("page_number", "") for m in metadatas],
            "modified_time": [m.get("modified_time", "") for m in metadatas],
            "source": [m.get("source", "") for m in metadatas],
            "text_preview": [(d or "")[:300] for d in documents],
        })

        st.caption(f"{len(df)} chunk(s) across {df['file_name'].nunique()} file(s) -- collection at `{db_path}`.")

        tab_table, tab_graph = st.tabs(["table", "graph"])

        with tab_table:
            col1, col2 = st.columns([2, 3])
            with col1:
                file_options = ["(all files)"] + sorted(df["file_name"].unique().tolist())
                selected_file = st.selectbox("Filter by file", file_options)
            with col2:
                search_text = st.text_input("Search in text preview", "")

            filtered = df
            if selected_file != "(all files)":
                filtered = filtered[filtered["file_name"] == selected_file]
            if search_text:
                filtered = filtered[filtered["text_preview"].str.contains(search_text, case=False, na=False)]

            st.caption(f"Showing {len(filtered)} of {len(df)} chunk(s)")
            st.dataframe(filtered, use_container_width=True, height=600)

        with tab_graph:
            st.caption(
                "2D projection of chunk embeddings via t-SNE. Points close "
                "together are semantically similar."
            )

            valid_rows = [
                (i, emb) for i, emb in enumerate(embeddings) if emb is not None
            ]
            if len(valid_rows) < 3:
                st.warning("Not enough embedded chunks to compute a t-SNE projection (need at least 3).")
            else:
                with st.spinner(f"Computing t-SNE over {len(valid_rows)} chunk(s)... this may take a while."):
                    from sklearn.manifold import TSNE
                    import numpy as np

                    indices = [i for i, _ in valid_rows]
                    vectors = np.array([emb for _, emb in valid_rows])

                    perplexity = min(30, max(5, len(vectors) // 3))
                    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
                    coords = tsne.fit_transform(vectors)

                plot_df = df.iloc[indices].copy()
                plot_df["x"] = coords[:, 0]
                plot_df["y"] = coords[:, 1]

                import plotly.express as px

                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    color="file_name",
                    hover_data=["file_name", "page_number", "text_preview"],
                    title=f"t-SNE projection of {len(plot_df)} chunks",
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-csv", metavar="PATH", help="Export the collection to a CSV file.")
    parser.add_argument("--web", action="store_true", help="Launch the interactive web viewer.")
    parser.add_argument("--port", type=int, default=8502, help="Port for --web (default: 8502).")
    parser.add_argument("--db-path", default=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    parser.add_argument("--render-web-ui", action="store_true", help=argparse.SUPPRESS,)

    args = parser.parse_args()

    if args.render_web_ui:
        _render_web_ui()
    elif args.web:
        launch_web(args.port)
    elif args.export_csv:
        export_csv(args.export_csv, args.db_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()