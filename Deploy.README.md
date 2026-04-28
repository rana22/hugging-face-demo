## 🚀 Deploying This App on Hugging Face Spaces
# ***hugging face will use README.md as deployment configuration***
This project can be deployed using **Hugging Face Spaces (Gradio)**. Follow the steps below for each team to deploy their own instance.

---

## 📋 Prerequisites

Each team should have:

- A Hugging Face account → https://huggingface.co/join  
- Git installed  
- This repository cloned locally  

---

## 🧭 Deployment Steps

### 1. Create a New Space

1. Go to: https://huggingface.co/spaces  
2. Click **“Create new Space”**
3. Configure:
   - **Owner**: Your team/org
   - **Space name**: e.g., `icdc-frontend-app`
   - **SDK**: `Gradio`
   - **Hardware**: CPU (default is fine)
   - **Visibility**: Public or Private  

Click **Create Space**

---

### 2. Clone the Space Repository

```bash
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cd <space-name>

### 3. Add Project Files

Copy these files into the Space repo:

```
app.py
requirements.txt
README.md
```

Make sure:

* `app.py` is at the root
* Your Gradio app is launched from `app.py`

---

### 4. Update `requirements.txt` (Important)

Use a version compatible with Hugging Face (Python 3.13):

```txt
numpy
pandas
scikit-learn
PyYAML
neo4j
python-dotenv
requests
matplotlib
openpyxl
uvicorn
fastapi

transformers
sentence-transformers
torch
fastapi
uvicorn

```

---

### 5. Commit and Push

```bash
git add .
git commit -m "Deploy app to Hugging Face Space"
git push
```

---

### 6. Wait for Build

* Go to your Space page
* Check the **Build Logs**
* Deployment usually takes 2–10 minutes

---

### 7. Access the App

Your app will be live at:

```
https://huggingface.co/spaces/<your-username>/<space-name>
```


