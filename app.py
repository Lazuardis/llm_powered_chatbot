import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:  # The app can still show the teaching UI before install.
    genai = None
    types = None


load_dotenv()


@dataclass(frozen=True)
class ModelOption:
    model_id: str
    label: str
    input_price_per_million: float
    output_price_per_million: float
    best_for: str


# Keep this table easy to update. Pricing changes over time, so the UI labels
# these as editable teaching estimates rather than permanent financial advice.
MODEL_OPTIONS = {
    "gemini-2.5-flash-lite": ModelOption(
        model_id="gemini-2.5-flash-lite",
        label="Gemini 2.5 Flash-Lite",
        input_price_per_million=0.10,
        output_price_per_million=0.40,
        best_for="Lowest-cost classroom demos and simple business chatbots.",
    ),
    "gemini-2.0-flash": ModelOption(
        model_id="gemini-2.0-flash",
        label="Gemini 2.0 Flash",
        input_price_per_million=0.10,
        output_price_per_million=0.40,
        best_for="Fast simple chatbots, if this model is enabled for your project.",
    ),
    "gemini-2.5-flash": ModelOption(
        model_id="gemini-2.5-flash",
        label="Gemini 2.5 Flash",
        input_price_per_million=0.30,
        output_price_per_million=2.50,
        best_for="Balanced quality and speed for more serious business tasks.",
    ),
    "gemini-2.5-pro": ModelOption(
        model_id="gemini-2.5-pro",
        label="Gemini 2.5 Pro",
        input_price_per_million=1.25,
        output_price_per_million=10.00,
        best_for="Higher-quality reasoning where accuracy matters more than cost.",
    ),
}


PROMPT_EXAMPLES = {
    "Marketing": {
        "instruction": "You are a marketing strategist helping small businesses create practical campaign ideas.",
        "prompt": "Create a simple campaign idea for a new coffee shop targeting university students.",
    },
    "Customer Support": {
        "instruction": "You are a friendly customer support assistant. Answer clearly and politely.",
        "prompt": "A customer says their order arrived late and asks for compensation. Draft a reply.",
    },
    "Human Resources": {
        "instruction": "You are an HR assistant helping managers write clear, respectful communication.",
        "prompt": "Write a short announcement inviting employees to a leadership training program.",
    },
    "Finance": {
        "instruction": "You are a finance analyst explaining business concepts to non-technical managers.",
        "prompt": "Explain why API usage costs grow when a chatbot gains more users.",
    },
    "Operations": {
        "instruction": "You are an operations consultant focused on efficiency and simple process improvement.",
        "prompt": "Suggest three ways a small retail store can use an AI chatbot to reduce repetitive work.",
    },
}


MODEL_CATALOG = [
    {
        "option_id": "google-gemini-2.5-flash-lite",
        "provider": "Google",
        "name": "Gemini 2.5 Flash-Lite",
        "status": "Available in this classroom app",
        "enabled": True,
        "model_id": "gemini-2.5-flash-lite",
    },
    {
        "option_id": "google-gemini-flash",
        "provider": "Google",
        "name": "Gemini Flash",
        "status": "Example alternative, not connected",
        "enabled": False,
        "model_id": None,
    },
    {
        "option_id": "openai-nano",
        "provider": "OpenAI",
        "name": "OpenAI Nano",
        "status": "Example alternative, not connected",
        "enabled": False,
        "model_id": None,
    },
    {
        "option_id": "anthropic-claude-haiku",
        "provider": "Anthropic",
        "name": "Claude Haiku",
        "status": "Example alternative, not connected",
        "enabled": False,
        "model_id": None,
    },
    {
        "option_id": "anthropic-claude-sonnet",
        "provider": "Anthropic",
        "name": "Claude Sonnet",
        "status": "Example alternative, not connected",
        "enabled": False,
        "model_id": None,
    },
]


def estimate_tokens(text: str) -> int:
    """Simple classroom-friendly fallback: about 4 characters per token."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def calculate_cost(tokens: int, price_per_million: float) -> float:
    return tokens / 1_000_000 * price_per_million


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def count_input_tokens(client, model_id: str, system_instruction: str, user_prompt: str) -> tuple[int, str]:
    combined_prompt = f"{system_instruction}\n\n{user_prompt}".strip()
    if client is None:
        return estimate_tokens(combined_prompt), "estimated"

    try:
        response = client.models.count_tokens(
            model=model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
        return int(response.total_tokens), "api"
    except Exception:
        return estimate_tokens(combined_prompt), "estimated"


def generate_response(client, model_id: str, system_instruction: str, user_prompt: str):
    if client is None:
        demo_text = (
            "Demo mode response: add your GEMINI_API_KEY to run a real API call. "
            "This preview keeps the classroom cost simulator usable without billing setup."
        )
        return {
            "text": demo_text,
            "input_tokens": estimate_tokens(f"{system_instruction}\n\n{user_prompt}"),
            "output_tokens": estimate_tokens(demo_text),
            "token_source": "estimated",
            "error": None,
        }

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        text = response.text or ""

        return {
            "text": text,
            "input_tokens": int(input_tokens) if input_tokens is not None else estimate_tokens(f"{system_instruction}\n\n{user_prompt}"),
            "output_tokens": int(output_tokens) if output_tokens is not None else estimate_tokens(text),
            "token_source": "api" if usage else "estimated",
            "error": None,
        }
    except Exception as exc:
        return {
            "text": "",
            "input_tokens": estimate_tokens(f"{system_instruction}\n\n{user_prompt}"),
            "output_tokens": 0,
            "token_source": "estimated",
            "error": str(exc),
        }


def format_usd(amount: float) -> str:
    if amount == 0:
        return "$0.00"
    if amount < 0.01:
        return f"${amount:.6f}"
    return f"${amount:,.2f}"


st.set_page_config(
    page_title="LLM API Cost Playground",
    page_icon="",
    layout="wide",
)

st.title("LLM API Cost & Prompt Playground")
st.caption("A simple chatbot lab for learning how model choice, prompts, tokens, and scale affect API cost.")

client = get_client()
api_ready = client is not None
package_ready = genai is not None

if not package_ready:
    st.warning("The Google GenAI package is not installed yet. Install dependencies with `pip install -r requirements.txt`.")
elif not api_ready:
    st.info("Demo mode: add `GEMINI_API_KEY` to a `.env` file to run real Gemini API calls.")

st.subheader("Model Selection")
model_dropdown_options = []
for catalog_model in MODEL_CATALOG:
    disabled_attr = "" if catalog_model["enabled"] else " disabled"
    selected_attr = " selected" if catalog_model["enabled"] else ""
    model_dropdown_options.append(
        f'<option{disabled_attr}{selected_attr}>'
        f'{catalog_model["provider"]} - {catalog_model["name"]}'
        "</option>"
    )

st.markdown(
    f"""
    <label for="model-catalog-select" style="font-size: 0.875rem; font-weight: 400;">
        Model name
    </label>
    <select id="model-catalog-select" style="
        width: 100%;
        min-height: 2.5rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: 0.45rem 0.75rem;
        font-size: 1rem;
        margin-top: 0.35rem;
        margin-bottom: 0.35rem;
    ">
        {''.join(model_dropdown_options)}
    </select>
    """,
    unsafe_allow_html=True,
)
st.caption("Only connected models can be selected. Other providers are visible as disabled alternatives.")
selected_model_id = "gemini-2.5-flash-lite"
model = MODEL_OPTIONS[selected_model_id]
model_is_available = True
st.caption(model.best_for)

with st.sidebar:
    st.header("Cost settings")
    pricing_mode = st.radio(
        "Pricing mode",
        ["Paid estimate", "Free-tier classroom demo"],
        help="Free-tier mode shows zero API cost but still tracks token usage.",
    )

    input_price = st.number_input(
        "Input price per 1M tokens",
        min_value=0.0,
        value=0.0 if pricing_mode == "Free-tier classroom demo" else model.input_price_per_million,
        step=0.01,
        format="%.4f",
    )
    output_price = st.number_input(
        "Output price per 1M tokens",
        min_value=0.0,
        value=0.0 if pricing_mode == "Free-tier classroom demo" else model.output_price_per_million,
        step=0.01,
        format="%.4f",
    )

    st.divider()
    st.header("Business scale")
    users = st.slider("Users", 1, 100_000, 1_000, step=100)
    messages_per_user = st.slider("Messages per user per day", 1, 100, 10)
    days_per_month = st.slider("Days per month", 1, 31, 30)


example_name = st.selectbox("Business example", list(PROMPT_EXAMPLES.keys()))
example = PROMPT_EXAMPLES[example_name]

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Model Instruction")
    system_instruction = st.text_area(
        "Role or behavior for the model",
        value=example["instruction"],
        height=130,
        help="This instruction is sent with the API call and counts as input tokens.",
    )

    st.subheader("User Prompt")
    user_prompt = st.text_area(
        "Message from the user",
        value=example["prompt"],
        height=180,
        help="This is the business question or task you send to the model.",
    )

    input_tokens, input_token_source = count_input_tokens(
        client,
        model.model_id,
        system_instruction,
        user_prompt,
    )
    input_cost = calculate_cost(input_tokens, input_price)

    metric_a, metric_b = st.columns(2)
    metric_a.metric("Input tokens", f"{input_tokens:,}", input_token_source)
    metric_b.metric("Estimated input cost", format_usd(input_cost))

    run_api_call = st.button(
        "Run API Call",
        type="primary",
        use_container_width=True,
        disabled=not model_is_available,
    )

with right_col:
    st.subheader("Model Response")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if run_api_call:
        if not user_prompt.strip():
            st.error("Write a user prompt before running the API call.")
        else:
            with st.spinner("Calling the model..."):
                st.session_state.last_result = generate_response(
                    client,
                    model.model_id,
                    system_instruction,
                    user_prompt,
                )

    result = st.session_state.last_result

    if result and result["error"]:
        st.error("The API call failed. The simulator is still showing estimated input cost.")
        st.code(result["error"], language="text")
    elif result:
        st.markdown(result["text"])
    else:
        st.write("Run the API call to see the response and output token cost.")

    output_tokens = result["output_tokens"] if result else 0
    if result and result["input_tokens"]:
        input_tokens = result["input_tokens"]
        input_cost = calculate_cost(input_tokens, input_price)

    output_cost = calculate_cost(output_tokens, output_price)
    total_call_cost = input_cost + output_cost

    metric_c, metric_d = st.columns(2)
    metric_c.metric("Output tokens", f"{output_tokens:,}", result["token_source"] if result else "pending")
    metric_d.metric("Estimated output cost", format_usd(output_cost))


st.divider()

cost_col, scale_col = st.columns([1, 1], gap="large")

with cost_col:
    st.subheader("Cost of This API Call")
    a, b, c = st.columns(3)
    a.metric("Input cost", format_usd(input_cost))
    b.metric("Output cost", format_usd(output_cost))
    c.metric("Total cost", format_usd(total_call_cost))

    st.code(
        "total_cost = (input_tokens / 1,000,000 x input_price) + "
        "(output_tokens / 1,000,000 x output_price)",
        language="text",
    )

with scale_col:
    st.subheader("Business Scaling Simulator")
    daily_calls = users * messages_per_user
    monthly_calls = daily_calls * days_per_month
    daily_cost = total_call_cost * daily_calls
    monthly_cost = total_call_cost * monthly_calls

    d, e, f = st.columns(3)
    d.metric("API calls/day", f"{daily_calls:,}")
    e.metric("Daily cost", format_usd(daily_cost))
    f.metric("Monthly cost", format_usd(monthly_cost))

    st.write(
        "The same chatbot feels cheap for one user, but the cost grows as more users "
        "send more messages and ask for longer answers."
    )

st.divider()
st.caption(
    "Pricing values are editable teaching estimates. Always check the provider pricing page before using this for budgeting."
)
