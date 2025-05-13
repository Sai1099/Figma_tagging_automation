import os
import json
import requests
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from transformers import pipeline, ViTImageProcessor, ViTModel
from dotenv import load_dotenv

load_dotenv()
FIGMA_TOKEN = os.getenv('FIGMA_TOKEN')
FIGMA_FILE_ID = os.getenv('FIGMA_FILE_ID')

HEADERS = {'X-Figma-Token': os.getenv(FIGMA_TOKEN)}
BASE_DIR = "data"
os.makedirs(BASE_DIR, exist_ok=True)

#finding components
def get_figma_components():
    url = f"https://api.figma.com/v1/files/{FIGMA_FILE_ID}"
    response = requests.get(url, headers=HEADERS)
    data = response.json()

    def find_components(node):
        components = []
        if node.get("type") in ["COMPONENT", "INSTANCE"]:
            components.append({
                "id": node["id"],
                "name": node.get("name", "")
            })
        for child in node.get("children", []):
            components.extend(find_components(child))
        return components

    components = find_components(data["document"])
    with open(f"{BASE_DIR}/components.json", "w") as f:
        json.dump(components, f, indent=2)
    print(f"Found {len(components)} components.")
    return components

#for tags
def add_ner_tags(components):
    nlp = pipeline("ner", model="dslim/bert-base-NER")
    for comp in components:
        comp["ner_tags"] = nlp(comp["name"])
    def convert_to_serializable(obj):
     if isinstance(obj, (np.float32, np.float64)):
         return float(obj)
     elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
     return obj
    with open(f"{BASE_DIR}/components_with_ner.json", "w") as f:
        json.dump(components, f, indent=2,default=convert_to_serializable)
    print("Added BERT NER tags.")



#images
def download_component_images(components):
    ids = ",".join([c["id"] for c in components])
    url = f"https://api.figma.com/v1/images/{FIGMA_FILE_ID}?ids={ids}&format=png"
    response = requests.get(url, headers=HEADERS)
    image_urls = response.json().get("images", {})

    img_dir = f"{BASE_DIR}/images"
    os.makedirs(img_dir, exist_ok=True)

    for comp in components:
        img_url = image_urls.get(comp["id"])
        if img_url:
            img_data = requests.get(img_url).content
            safe_id = comp["id"].replace(":", "_").replace(";", "_")
            comp["safe_id"] = safe_id 
            with open(f"{img_dir}/{safe_id}.png", "wb") as f:
                f.write(img_data)
    def convert_to_serializable(obj):
     if isinstance(obj, (np.float32, np.float64)):
         return float(obj)
     elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
     return obj
    with open(f"{BASE_DIR}/components_with_safe_ids.json", "w") as f:
        json.dump(components, f, indent=2,default=convert_to_serializable)

    print("Downloaded component images")

#extract visual features 
def extract_and_cluster_visual_features():
    processor = ViTImageProcessor.from_pretrained("facebook/dino-vits8")
    model = ViTModel.from_pretrained("facebook/dino-vits8")

    features = []
    ids = []
    img_dir = f"{BASE_DIR}/images"

    for file in os.listdir(img_dir):
        if file.endswith(".png"):
            img = Image.open(f"{img_dir}/{file}").convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            features.append(vector)
            ids.append(file.replace(".png", ""))

    kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
    with open(f"{BASE_DIR}/components_with_ner.json", "r") as f:
        components = json.load(f)

    for comp in components:
        if comp["id"] in ids:
            index = ids.index(comp["id"])
            comp["cluster"] = int(kmeans.labels_[index])

    with open(f"{BASE_DIR}/components_final.json", "w") as f:
        json.dump(components, f, indent=2)
    print("Clustered visual features.")
    return components

#predefined
PREDEFINED_LABELS = {
    "button_cta": ["login", "signup", "cta", "submit", "get started"],
    "login_button": ["login", "sign in"],
    "navbar": ["home", "menu", "navigation", "nav", "about"],
    "footer": ["footer", "contact", "help"]
}

def match_predefined_label(tags, cluster_id=None):
    tag_words = [tag['word'].lower() for tag in tags]
    for label, keywords in PREDEFINED_LABELS.items():
        for keyword in keywords:
            if any(keyword in word for word in tag_words):
                return label
    return "unlabeled"

#comments
def comment_on_components(components):
    COMMENT_URL = f"https://api.figma.com/v1/files/{FIGMA_FILE_ID}/comments"

    for comp in components:
        name = comp["name"]
        cluster = comp.get("cluster", "N/A")
        ner_tags = comp.get("ner_tags", [])
        label = match_predefined_label(ner_tags, cluster)

        comp["matched_label"] = label  # Save for inspection/debugging

        payload = {
            "message": (
                f"Detected Label: {label}\n"
                f"Component Name: {name}\n"
                f"NER Tags: {', '.join([t['word'] for t in ner_tags])}\n"
                f"Cluster Group: {cluster}"
            ),
            "client_meta": {
                "node_id": comp["id"],
                "node_offset": {"x": 0, "y": 0}
            }
        }

        res = requests.post(COMMENT_URL, headers=HEADERS, json=payload)
        print(f"Commented on '{name}' → '{label}' — Status: {res.status_code}")

    with open(f"{BASE_DIR}/components_labeled.json", "w") as f:
        json.dump(components, f, indent=2)



def run_pipeline():
    components = get_figma_components()
    add_ner_tags(components)
    download_component_images(components)
    clustered_components = extract_and_cluster_visual_features()
    comment_on_components(clustered_components)

if __name__ == "__main__":
    run_pipeline()
