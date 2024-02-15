from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Load pre-trained model and tokenizer
model_name = "thilina/mt5-sinhalese-english"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Function to translate English text to Sinhala
def translate_to_sinhala(english_text):
    inputs = tokenizer.encode(english_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Example usage
english_text = "Hello we are going to reach out to you as soon as possible."
sinhala_translation = translate_to_sinhala(english_text)
print("Sinhala Translation:", sinhala_translation)
