MODEL_NAME = "google/flan-t5-base"


def load_generator():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def convert_tone(text, tone, generator):
    tokenizer, model = generator
    prompt = (
        f"Rewrite the text below in a {tone} tone. Keep the same meaning. "
        f"Return only the rewritten text.\n\nText: {text.strip()}"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        do_sample=False,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def choose_tone():
    tone_map = {
        "1": "professional",
        "2": "friendly",
        "3": "informal",
        "4": "formal",
    }

    print("\nSelect tone:")
    print("1. Professional")
    print("2. Friendly")
    print("3. Informal")
    print("4. Formal")

    choice = input("Enter choice (1-4): ").strip()
    return tone_map.get(choice, "professional")


def main():
    print("Antigravity Tone Transformer\n")

    text = input("Enter your text:\n> ").strip()
    if not text:
        print("Please enter valid text.")
        return

    tone = choose_tone()

    try:
        generator = load_generator()
        output = convert_tone(text, tone, generator)
    except ImportError:
        print("\nUnable to transform the text.")
        print("The required Python packages are not installed.")
        print("Run this command first:")
        print("pip install -r requirements.txt")
        return
    except Exception as error:
        print("\nUnable to transform the text.")
        print("Make sure dependencies are installed with:")
        print("pip install -r requirements.txt")
        print(f"\nDetails: {error}")
        return

    print("\nTransformed Text:\n")
    print(output)


if __name__ == "__main__":
    main()
