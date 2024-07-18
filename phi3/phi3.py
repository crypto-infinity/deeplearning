import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():

    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    while True:
        
        user_input = input("Cosa mi vuoi chiedere? (scrivi 'no' per terminare): ")
        if user_input.lower() == "no":
            break

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_input},
        ]

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)
        print(output[0]['generated_text'])

if __name__ == "__main__":
    main()
