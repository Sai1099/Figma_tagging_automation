# Figma Tagging Automation

This project automates the process of analyzing, tagging, and clustering Figma components using **NLP** and **Computer Vision** techniques. It integrates the Figma API, Hugging Face Transformers, and scikit-learn to extract structured information from Figma design files and provides automated feedback by posting comments directly on components inside Figma.

---

## Features

- **Fetch Components**: Extracts all `COMPONENT` and `INSTANCE` nodes from a Figma file.
- **NER Tagging**: Uses a BERT-based Named Entity Recognition (NER) model to generate semantic tags for component names.
- **Image Download**: Fetches and saves component images locally via the Figma Images API.
- **Visual Feature Extraction**: Applies Vision Transformer (ViT) models to extract embeddings from component images.
- **Clustering**: Groups visually similar components using K-Means clustering.
- **Predefined Label Matching**: Maps semantic tags to common UI labels (e.g., buttons, navbars, footers).
- **Automated Comments**: Posts labels, NER tags, and cluster details as comments on corresponding Figma components.

---

## Tech Stack

- **Python 3.8+**
- **Libraries**:
  - `transformers` (NER & Vision Transformer models)
  - `scikit-learn` (K-Means clustering)
  - `PIL` (image processing)
  - `numpy`
  - `requests`
  - `dotenv`
- **APIs**:
  - Figma REST API

---



