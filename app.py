import streamlit as st
import pandas as pd
import os
from labeling.image_labeler import label_images
from labeling.text_labeler import label_text_df

st.set_page_config(page_title="Auto Labeler", layout="wide")
st.title("📌 Auto Labeler - Label Your Images and Text Automatically")

task = st.sidebar.selectbox("Select Task", ["Image Labeling", "Text Labeling (CSV)"])

if task == "Image Labeling":
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if uploaded_files:
        image_paths = []
        save_dir = "uploaded_images"
        os.makedirs(save_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            image_paths.append(file_path)

        st.success(f"{len(image_paths)} image(s) uploaded.")

        if st.button("Label Images"):
            with st.spinner("Labeling in progress..."):
                results = label_images(image_paths)

                # Display results
                for filename, label in results:
                    st.write(f"📷 **{filename}** → 🏷️ {label}")

                # Create downloadable DataFrame
                df = pd.DataFrame(results, columns=["Image", "Predicted Label"])
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Labeled Results (CSV)", csv, "labeled_images.csv", "text/csv")

elif task == "Text Labeling (CSV)":
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("📄 Uploaded CSV Preview:")
        st.dataframe(df.head())

        text_column = st.selectbox("Select the column containing text", df.columns)
        labels_input = st.text_input("Enter possible labels (comma separated)", "positive, negative, neutral")
        labels = [label.strip() for label in labels_input.split(",")]

        if st.button("Label Text"):
            with st.spinner("Labeling text..."):
                labeled_df = label_text_df(df, text_column, labels)
                st.success("✅ Text labeling completed!")
                st.dataframe(labeled_df)

                csv = labeled_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Labeled CSV", csv, "labeled_output.csv", "text/csv")
