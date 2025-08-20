# 🎨 Figma Tagging Automation

This project automates **component tagging and labeling** for Figma designs using **AI + clustering**.  
It extracts components from a Figma file, applies **NER-based tags**, clusters them visually with **Vision Transformers**, and posts **automated comments** back into Figma for easier design review and organization.  

---

## ⚡ Features
- ✅ Fetches all **components & instances** from a Figma file.  
- ✅ Adds **NER tags** to component names using BERT (`dslim/bert-base-NER`).  
- ✅ Downloads **component images** for analysis.  
- ✅ Extracts **visual features** with `facebook/dino-vits8` (ViT).  
- ✅ Clusters components using **KMeans** into visual groups.  
- ✅ Matches **predefined labels** (e.g., `button_cta`, `navbar`, `footer`).  
- ✅ Automatically **comments on components in Figma** with detected tags & labels.  

---

## 🏗️ Architecture / Workflow
1. **Scrape components** → Get all components & instances from Figma.  
2. **NER tagging** → Extract meaningful entity tags from component names.  
3. **Image download** → Download Figma component images.  
4. **Visual clustering** → Extract embeddings from ViT and cluster via KMeans.  
5. **Predefined label mapping** → Map NER tags to predefined UI labels.  
6. **Comment back to Figma** → Add structured labels directly as comments.  

