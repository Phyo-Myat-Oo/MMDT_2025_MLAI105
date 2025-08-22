import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_excel("../result_summary.xlsx")
# Create a fixed color map for all models
model_list = df["Model description"].unique().tolist()
palette = px.colors.qualitative.Plotly  # or "Set2", "Dark2", etc.

color_map = {m: palette[i % len(palette)] for i, m in enumerate(model_list)}

# ---------------------------
# Convert metrics to clean float
# ---------------------------
metrics = ["Precision", "Recall", "F1"]
for m in metrics:
    df[m] = (
        df[m]
        .astype(str)          # force string
        .str.strip()          # remove spaces
        .replace("", "0")     # handle empty strings
    )
    df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0).astype(float).round(3)

# ---------------------------
# Dataset Info
# ---------------------------
total_sentences = 71711
total_tokens = 2620226
unique_words = 25505
label_classes = ["LOCATION", "DATE", "TIME", "OUTSIDE"]
tagging_scheme = "BIO format (Begin, Inside, Outside)"

label_distribution = {
    "O": 2483849,
    "B-LOC": 54918,
    "I-LOC": 40131,
    "I-DATE": 21456,
    "B-DATE": 13293,
    "I-TIME": 3681,
    "B-TIME": 2898,
}

# ---------------------------
# Streamlit Layout
# ---------------------------
st.set_page_config(page_title="Burmese NER Dashboard", layout="wide")
st.title("📊 Burmese Named Entity Recognition (NER) Results Dashboard")

# Dataset Info Section
st.header("📘 Dataset Information")
st.markdown(f"""
- **Total sentences**: {total_sentences:,}  
- **Total tokens in corpus**: {total_tokens:,}  
- **Unique words in corpus**: {unique_words:,}  
- **Label classes**: {", ".join(label_classes)}  
- **Label tagging scheme**: {tagging_scheme}  
""")

# Merged LOC / DATE / TIME distribution
st.subheader("📌 Entity Distribution (Merged)")
merged_distribution = {
    "LOCATION": label_distribution["B-LOC"] + label_distribution["I-LOC"],
    "DATE": label_distribution["B-DATE"] + label_distribution["I-DATE"],
    "TIME": label_distribution["B-TIME"] + label_distribution["I-TIME"],
}
merged_df = pd.DataFrame(list(merged_distribution.items()), columns=["Entity Type", "Count"])
fig_merged = px.bar(
    merged_df, x="Entity Type", y="Count", text="Count",
    title="NER Label Distribution (Merged)"
)
fig_merged.update_traces(texttemplate='%{text:,}', textposition='outside')
st.plotly_chart(fig_merged, use_container_width=True)

# ---------------------------
# Model Performance Visualization
# ---------------------------
st.header("📊 Model Performance Visualization")

# Sidebar filters
st.sidebar.header("Filters")
selected_model = st.sidebar.multiselect(
    "Select Model(s)", df["Model description"].unique(),
    default=df["Model description"].unique()
)
selected_entity = st.sidebar.multiselect(
    "Select Entity", df["Entity"].unique(),
    default=df["Entity"].unique()
)

filtered_df = df[
    (df["Model description"].isin(selected_model)) &
    (df["Entity"].isin(selected_entity))
]

# Helper function to plot individual metric bar charts in descending order
def plot_metric_bar_desc(df, metric_name, title):
    df = df.copy()
    df[metric_name] = df[metric_name].astype(float)
    
    # Compute mean metric per entity for ordering
    entity_order = df.groupby("Entity")[metric_name].mean().sort_values(ascending=True).index.tolist()
    
    fig = px.bar(
    df,
    x="Entity",
    y=metric_name,
    color="Model description",             # <- column name
    color_discrete_map=color_map,          # <- fixed color mapping
    barmode="group",
    title=title,
    hover_data=["Model"],
    category_orders={"Entity": entity_order}  # force descending order
)

    # fig = px.bar(
    #     df,
    #     x="Entity",
    #     y=metric_name,
    #     color=color_map,
    #     barmode="group",
    #     title=title,
    #     hover_data=["Model"],
    #     category_orders={"Entity": entity_order}  # force descending order
    # )
    fig.update_yaxes(range=[0, 1])
    return fig

# Precision Chart
st.subheader("Entity-wise Precision by Model")
st.plotly_chart(plot_metric_bar_desc(filtered_df, "Precision", "Precision Comparison"), use_container_width=True)

# Recall Chart
st.subheader("Entity-wise Recall by Model")
st.plotly_chart(plot_metric_bar_desc(filtered_df, "Recall", "Recall Comparison"), use_container_width=True)

# F1 Score Chart
st.subheader("Entity-wise F1 Score by Model")
st.plotly_chart(plot_metric_bar_desc(filtered_df, "F1", "F1 Score Comparison"), use_container_width=True)


import pandas as pd
import plotly.express as px
import streamlit as st

# Exclude the 'O' entity
df_filtered = df[df["Entity"] != "O"]

# Compute average F1 per model (ignoring 'O')
df_avg = df_filtered.groupby(
    ["Model description", "Model", "Environment", "Training Time(Sec)"],
    as_index=False
).agg({"F1": "mean"})

# Plot with averaged F1 (without 'O')
st.subheader("Training Time vs Average F1 Score (excluding 'O')")
fig_time = px.scatter(
    df_avg,
    x="Training Time(Sec)",
    y="F1",
    color="Model description", 
    color_discrete_map=color_map,   
    hover_data=["Model", "Environment"],
    title="Training Time vs Average F1 Score (excluding 'O')"
)
fig_time.update_yaxes(range=[0, 1])
st.plotly_chart(fig_time, use_container_width=True)

# ---------------------------
# Environment Usage
# ---------------------------
st.subheader("CPU vs GPU Environment Usage")
fig_env = px.histogram(df, x="Environment", color="Model description", title="Environment Distribution")
st.plotly_chart(fig_env, use_container_width=True)


# from PIL import Image

# # Load the confusion matrix image
# cm_image = Image.open("../confusion_matrix.png") 

# # Display in Streamlit
# st.subheader("Confusion Matrix for Best Model: DistilBERT + CRF")
# st.image(cm_image, caption="Confusion Matrix", use_container_width=True)
from PIL import Image

# Load image
cm_image = Image.open("../confusion_matrix.png")

# Resize image (e.g., width=800, maintain aspect ratio)
base_width = 800
w_percent = base_width / float(cm_image.size[0])
h_size = int(float(cm_image.size[1]) * w_percent)
cm_image_resized = cm_image.resize((base_width, h_size), Image.Resampling.LANCZOS)

# Display in Streamlit
st.subheader("Confusion Matrix for best model: DistilBERT + CRF")
st.image(cm_image_resized, caption="Confusion Matrix", use_container_width=True)


import pandas as pd
import streamlit as st

# Load Error Analysis Sheet
df_error = pd.read_excel("../result_summary.xlsx", sheet_name="predictedVsactual")
df_error = df_error.fillna("").astype(str)

# Streamlit Layout
st.header("📝 Error Analysis Table (Token-level) on random sentences")

# Sentence filter
sentence_ids = df_error["Sentence ID"].unique()
selected_sentence = st.selectbox("Select Sentence ID", sentence_ids)

df_sent = df_error[df_error["Sentence ID"] == selected_sentence].copy()

# Highlight mismatches
def highlight_mismatch(row):
    # Make a style list with length = number of columns
    style_list = [""] * len(row)
    if row["True NER"] != row["Predicted NER"]:
        style_list[df_sent.columns.get_loc("Predicted NER")] = "background-color: #ff9999; font-weight:bold;"
    return style_list

# Display table with highlighting
st.dataframe(df_sent.style.apply(highlight_mismatch, axis=1))


# # ---------------------------
# # Streamlit Layout
# # ---------------------------
# st.header("📝 Error Analysis on Random Sentences")

# sentence_ids = df_error["Sentence ID"].unique()

# for sent_id in sentence_ids:
#     st.subheader(f"Sentence {sent_id}")
#     df_sent = df_error[df_error["Sentence ID"] == sent_id]

#     tokens = df_sent["Token"].tolist()
#     true_labels = df_sent["True NER"].tolist()
#     pred_labels = df_sent["Predicted NER"].tolist()

#     # Build HTML for highlighting mismatches
#     token_line = " ".join(tokens)
#     true_line = " ".join(true_labels)
    
#     pred_line_parts = []
#     for t, p in zip(true_labels, pred_labels):
#         if t != p:
#             # Highlight mismatch in red
#             pred_line_parts.append(f"<span style='color:red;font-weight:bold'>{p}</span>")
#         else:
#             pred_line_parts.append(p)
#     pred_line = " ".join(pred_line_parts)

#     # Display lines
#     st.markdown(f"**Tokens:** {token_line}")
#     st.markdown(f"**True NER:** {true_line}")
#     st.markdown(f"**Predicted NER:** {pred_line}", unsafe_allow_html=True)

#     st.markdown("---")  # separator between sentences
