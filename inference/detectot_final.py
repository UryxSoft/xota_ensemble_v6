import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from tokenizers import normalizers
from tokenizers.normalizers import Sequence, Replace, Strip, NFKC
from tokenizers import Regex
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1_path = "/content/modernbert.bin"
model2_path = "https://huggingface.co/mihalykiss/modernbert_2/resolve/main/Model_groups_3class_seed12"
model3_path = "https://huggingface.co/mihalykiss/modernbert_2/resolve/main/Model_groups_3class_seed22"
model4_path = "https://huggingface.co/mihalykiss/ModernBERT-MGT/resolve/main/Model_groups_41class_seed44__new"

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

model_1 = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=41)
model_1.load_state_dict(torch.load(model1_path, map_location=device))
model_1.to(device).eval()

model_2 = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=41)
model_2.load_state_dict(torch.hub.load_state_dict_from_url(model2_path, map_location=device))
model_2.to(device).eval()

model_3 = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=41)
model_3.load_state_dict(torch.hub.load_state_dict_from_url(model3_path, map_location=device))
model_3.to(device).eval()

model_4 = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=41)
model_4.load_state_dict(torch.hub.load_state_dict_from_url(model4_path, map_location=device))
model_4.to(device).eval()



label_mapping = {
    0: '13B', 1: '30B', 2: '65B', 3: '7B', 4: 'GLM130B', 5: 'bloom_7b',
    6: 'bloomz', 7: 'cohere', 8: 'davinci', 9: 'dolly', 10: 'dolly-v2-12b',
    11: 'flan_t5_base', 12: 'flan_t5_large', 13: 'flan_t5_small',
    14: 'flan_t5_xl', 15: 'flan_t5_xxl', 16: 'gemma-7b-it', 17: 'gemma2-9b-it',
    18: 'gpt-3.5-turbo', 19: 'gpt-35', 20: 'gpt4', 21: 'gpt4o',
    22: 'gpt_j', 23: 'gpt_neox', 24: 'human', 25: 'llama3-70b', 26: 'llama3-8b',
    27: 'mixtral-8x7b', 28: 'opt_1.3b', 29: 'opt_125m', 30: 'opt_13b',
    31: 'opt_2.7b', 32: 'opt_30b', 33: 'opt_350m', 34: 'opt_6.7b',
    35: 'opt_iml_30b', 36: 'opt_iml_max_1.3b', 37: 't0_11b', 38: 't0_3b',
    39: 'text-davinci-002', 40: 'text-davinci-003'
}

def clean_text(text: str) -> str:
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([,.;:?!])', r'\1', text)
    return text

newline_to_space = Replace(Regex(r'\s*\n\s*'), " ")
join_hyphen_break = Replace(Regex(r'(\w+)[--]\s*\n\s*(\w+)'), r"\1\2")
tokenizer.backend_tokenizer.normalizer = Sequence([
    tokenizer.backend_tokenizer.normalizer,
    join_hyphen_break,
    newline_to_space,
    Strip()
])


def classify_text(text):
    """
    Classifies the text and generates a plot of the human vs AI probability.
    Returns both the result message and the plot figure.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        return "", None

    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits_1 = model_1(**inputs).logits
        logits_2 = model_2(**inputs).logits
        logits_3 = model_3(**inputs).logits
        logits_4 = model_4(**inputs).logits

        softmax_1 = torch.softmax(logits_1, dim=1)
        softmax_2 = torch.softmax(logits_2, dim=1)
        softmax_3 = torch.softmax(logits_3, dim=1)
        softmax_4 = torch.softmax(logits_4, dim=1)

        averaged_probabilities = (softmax_1 + softmax_2 + softmax_3 + softmax_4) / 4
        probabilities = averaged_probabilities[0]

    human_prob = probabilities[24].item()
    ai_probs_clone = probabilities.clone()
    ai_probs_clone[24] = 0
    ai_total_prob = ai_probs_clone.sum().item()

    total_decision_prob = human_prob + ai_total_prob
    human_percentage = (human_prob / total_decision_prob) * 100
    ai_percentage = (ai_total_prob / total_decision_prob) * 100

    ai_argmax_index = torch.argmax(ai_probs_clone).item()
    ai_argmax_model = label_mapping[ai_argmax_index]

    if human_percentage > ai_percentage:
        result_message = (
            f"**The text is** <span class='highlight-human'>**{human_percentage:.2f}%** likely <b>Human written</b>.</span>"
        )
    else:
        result_message = (
            f"**The text is** <span class='highlight-ai'>**{ai_percentage:.2f}%** likely <b>AI generated</b>.</span>\n\n"
        )

    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust figure size for better layout

    categories = ['Human', 'AI']
    probabilities_for_plot = [human_percentage, ai_percentage]

    bars = ax.bar(categories, probabilities_for_plot, color=['#4CAF50', '#FF5733'], alpha=0.8)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Human vs AI Probability', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center')

    ax.set_ylim(0, 100)
    plt.tight_layout()

    return result_message, fig
"""
with iface:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    text_input = gr.Textbox(label="", placeholder="Type or paste your content here...", elem_id="text_input_box", lines=5)
    result_output = gr.Markdown("", elem_id="result_output_box")
    plot_output = gr.Plot(label="Model Probability Distribution")

    text_input.change(classify_text, inputs=text_input, outputs=[result_output, plot_output])

    with gr.Tab("AI text examples"):
        gr.Examples(AI_texts, inputs=text_input)
    with gr.Tab("Human text examples"):
        gr.Examples(Human_texts, inputs=text_input)
    gr.Markdown(bottom_text, elem_id="bottom_text")

iface.launch(share=True)
"""
