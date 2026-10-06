# Run in Google Colab

Use these cells in Colab.

## 1. Clone the project

```python
!git clone https://github.com/AnkitYuva/RAG-App.git
%cd RAG-App
```

## 2. Install minimum requirements

```python
!pip install -r requirements.txt
```

## 3. Add your OpenRouter API key

```python
import os
os.environ["OPENROUTER_API_KEY"] = "PASTE_YOUR_OPENROUTER_API_KEY_HERE"
```

## 4. Run the Gradio app

```python
!python app.py
```

Open the public Gradio link printed by Colab.

Minimum dependencies:

- `gradio`
- `google-generativeai`
- `pypdf`
- `python-dotenv`
