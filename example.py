from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline


model = AutoModelForCausalLM.from_pretrained("lightonai/RITA_s", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("lightonai/RITA_s")

rita_gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
sequences = rita_gen("MAB", max_length=20, do_sample=True, top_k=950, repetition_penalty=1.2, 
                     num_return_sequences=2, eos_token_id=2)
for seq in sequences:
    print(f"seq: {seq['generated_text'].replace(' ', '')}")
