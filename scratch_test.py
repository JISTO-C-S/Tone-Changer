from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

text = "Hey bro, send the file quickly"
tone = "professional"

prompts = [
    f"Here is an informal sentence: '{text}'. Rewrite it to be {tone}.",
    f"Translate from informal to {tone}: {text}",
    f"Rewrite the following text in a {tone} tone while preserving meaning: {text}"
]

for p in prompts:
    inputs = tokenizer(p, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {p}")
    print(f"Result: {res}\n")
