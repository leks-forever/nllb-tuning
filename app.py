import gradio as gr
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = NllbTokenizer.from_pretrained("leks-forever/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("leks-forever/nllb-200-distilled-600M")
model = model.to(device)

def translate(text, src_lang='lez_Cyrl', tgt_lang='rus_Cyrl', a=32, b=3, max_input_length=1024, num_beams=1, **kwargs):
      global tokenizer

      if src_lang in language_codes:
        src_lang = language_codes[src_lang]

      if tgt_lang in language_codes: 
        tgt_lang = language_codes[tgt_lang]

      tokenizer.src_lang = src_lang
      tokenizer.tgt_lang = tgt_lang
      
      inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length)
      result = model.generate(
          **inputs.to(model.device),
          forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
          max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
          num_beams=num_beams,
          **kwargs
      )
      return tokenizer.batch_decode(result, skip_special_tokens=True)[0]

import gradio as gr

src_lang, tgt_lang = "Русский", "Лезги чlал"

interface = {
    "Лезги чlал": {
        "title": 'Лезги-урус чlалар', 
        'placeholder': 'кхьих лезги чlалал', 
        'translate_btn': 'Элкъурун', 
        'lang_swap_btn': 'чlал Дегишрун',
    }, 
    "Русский": {
        "title": 'русско-лезгинский переводчик', 
        'placeholder': 'введите текст на русском для перевода', 
        'translate_btn': 'перевести', 
        'lang_swap_btn': 'сменить язык',
    }, 
}

language_codes = {
    "Русский": "rus_Cyrl",
    "Лезги чlал": "lez_Cyrl",
}

def swap_languages(src_lang, tgt_lang, src_text):
    # Swap languages and update the relevant UI components (placeholder, button labels)
    new_src_lang = tgt_lang
    new_tgt_lang = src_lang

    # Update components' properties dynamically
    return (
        new_src_lang,
        new_tgt_lang,
        gr.Textbox(placeholder=interface[new_src_lang]['placeholder'], value=''),  # Clear and update placeholder
        gr.Markdown(f"# {interface[tgt_lang]['title']}"),
        gr.Button(value=interface[new_src_lang]['translate_btn']),  # Update translate button label
        gr.Button(value=interface[new_src_lang]['lang_swap_btn'])  # Update swap button label
    )

with gr.Blocks() as demo:
    title = gr.Markdown(f"# {interface[src_lang]['title']}")

    with gr.Row():
        with gr.Column():
            src_text = gr.Textbox(label='', placeholder=interface[src_lang]['placeholder'])
        with gr.Column():
            tgt_text = gr.Textbox(label='', interactive=False)

    src_lang_state = gr.State(value=src_lang)
    tgt_lang_state = gr.State(value=tgt_lang)

    translate_btn = gr.Button(interface[src_lang]['translate_btn'])
    swap_button = gr.Button(interface[src_lang]['lang_swap_btn'])

    translate_btn.click(
        fn=translate,
        inputs=[src_text, src_lang_state, tgt_lang_state], 
        outputs=tgt_text
    )
    swap_button.click(
        fn=swap_languages,
        inputs=[src_lang_state, tgt_lang_state, src_text], 
        outputs=[src_lang_state, tgt_lang_state, src_text, title, translate_btn, swap_button]  # Update states and components
    )

if __name__ == "__main__":
    demo.launch()
