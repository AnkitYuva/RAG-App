# 🚀 Exercise 3: Deployment Guide (10 Marks)
## Deploying the RAG Application from GitHub to Cloud

This guide provides step-by-step instructions to initialize Git, push this repository to GitHub, and deploy it to a live cloud environment (Hugging Face Spaces or Render) with zero cost.

---

## 📋 Overview of Deployment Architecture

```
   ┌─────────────────────────────────────────────────────────────┐
   │                     LOCAL LAPTOP                            │
   │  git add . -> git commit -> git push origin main            │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                     GITHUB REPOSITORY                       │
   │       https://github.com/<your-username>/Sem1RAG            │
   └──────────────┬───────────────────────────────┬──────────────┘
                  │ (Option A: Direct Sync)       │ (Option B: Docker / Web Service)
                  ▼                               ▼
   ┌──────────────────────────────┐ ┌────────────────────────────┐
   │    HUGGING FACE SPACES       │ │      RENDER / RAILWAY      │
   │  - Native Gradio Hosting     │ │  - Container / Python App  │
   │  - Free 24/7 HTTPS URL       │ │  - Automatic Build via Git │
   │  - 1-Click Sync from GitHub  │ │  - Port Binding from $PORT │
   └──────────────────────────────┘ └────────────────────────────┘
```

---

## 🛠️ Step 1: Initialize Git and Push to GitHub

Run these commands inside the project folder (`c:\Users\ankit\Desktop\Sem1RAG`):

### 1.1 Check Git Status & Initialize
Open PowerShell / Terminal in `c:\Users\ankit\Desktop\Sem1RAG`:

```powershell
# 1. Initialize git in this directory
git init

# 2. Stage all project files (ignoring venv and .env via .gitignore)
git add .

# 3. Commit the code
git commit -m "feat: complete RAG application with professional prompt design and deployment"
```

### 1.2 Create a New GitHub Repository
1. Log in to [GitHub](https://github.com).
2. Click **New Repository** (`+` in top right).
3. Name it: `Sem1RAG` (or `rag-agentic-ai`).
4. Set visibility to **Public** (recommended for grading/demo) or **Private**.
5. Do **NOT** initialize with README or .gitignore (we already have them).
6. Click **Create repository**.

### 1.3 Link and Push
Run the commands shown by GitHub:

```powershell
# Rename branch to main
git branch -M main

# Link to your remote GitHub repository (replace with your actual URL)
git remote add origin https://github.com/<your-username>/Sem1RAG.git

# Push the code
git push -u origin main
```

---

## 🌐 Step 2: Deploy to Cloud

### Option A (Recommended): 1-Click Deploy on Hugging Face Spaces (Free & Instant)
Hugging Face Spaces natively hosts Gradio apps with zero configuration:

1. Go to [Hugging Face](https://huggingface.co) and sign in (create a free account if needed).
2. Click on your profile icon (top right) → **New Space**.
3. Fill in the details:
   - **Space name**: `rag-document-assistant`
   - **License**: `apache-2.0` (or `mit`)
   - **Space SDK**: Select **Gradio**
   - **Space hardware**: `Free CPU (basic)`
4. Click **Create Space**.
5. **Connect your GitHub Repo**:
   - Go to the **Settings** tab of your Space.
   - Under **Repository**, link your GitHub repo (`<your-username>/Sem1RAG`), OR simply clone and push using HF git:
     ```powershell
     git remote add space https://huggingface.co/spaces/<your-username>/rag-document-assistant
     git push space main
     ```
6. **Set your API Key Secret**:
   - In your Hugging Face Space, click **Settings** → **Variables and secrets**.
   - Under **Secrets**, click **New secret**:
     - Key: `OPENROUTER_API_KEY`
     - Value: `<your-api-key>`
7. Your app builds in ~60 seconds and gives you a permanent, live HTTPS URL:
   `https://huggingface.co/spaces/<your-username>/rag-document-assistant`!

---

### Option B: Deploy to Render.com (Web Service)
1. Go to [Render.com](https://render.com) and sign in with GitHub.
2. Click **New +** → **Web Service**.
3. Select your GitHub repository (`Sem1RAG`).
4. Configure:
   - **Name**: `sem1-rag-assistant`
   - **Region**: Choose closest (e.g. Singapore / Frankfurt / Oregon)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. Under **Environment Variables**:
   - Add `OPENROUTER_API_KEY` = `your_openrouter_api_key_here`
   - Add `GRADIO_SERVER_NAME` = `0.0.0.0`
6. Click **Deploy Web Service**.
7. Render will build and launch your application at:
   `https://sem1-rag-assistant.onrender.com`

---

## 💻 Step 3: Local Demonstration on Laptop (Class Requirement)

The instructions state:
> *"You must demonstrate your work on your own laptop during the class and show me the execution before the IAT exam."*

### Running Locally:
In your project directory:
```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Run the app
python app.py
```
Open your browser to:
`http://127.0.0.1:7860`

### Showing the Examiner:
1. **Show Local Execution**: Point your browser to `http://127.0.0.1:7860`.
2. **Show GitHub Repo**: Show `https://github.com/<your-username>/Sem1RAG` with recent commit history.
3. **Show Deployed Cloud URL**: Open your live Hugging Face or Render link on another tab (or your mobile phone) to prove the application is deployed and operational in a cloud environment!
