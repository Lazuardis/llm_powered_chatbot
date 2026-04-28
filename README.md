# LLM API Cost & Prompt Playground

A simple Streamlit chatbot lab for non-technical business students.

The app demonstrates:

- model selection
- visible but disabled examples of alternative providers and models
- model instruction / role prompting
- user prompts
- input and output token counting
- estimated API cost per call
- monthly business cost simulation as users and usage grow

The default model is `gemini-2.5-flash-lite`, which is a good classroom-friendly starting point.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Then edit `.env` and add your Gemini API key:

```text
GEMINI_API_KEY=your_real_key
```

Keep `.env` private. Do not commit real API keys to GitHub; `.env.example` should contain placeholders only.

## Run

```powershell
streamlit run app.py
```

If no API key is provided, the app runs in demo mode with estimated tokens and a placeholder response.

## Teaching Note

The main formula is:

```text
total_cost = (input_tokens / 1,000,000 x input_price) + (output_tokens / 1,000,000 x output_price)
```

Pricing changes over time, so the sidebar keeps input and output prices editable.
