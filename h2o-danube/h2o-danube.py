import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="h2oai/h2o-danube2-1.8b-chat",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# We use the HF Tokenizer chat template to format each message
# https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {"role": "user", "content": "Why is drinking water so healthy?"},
]
prompt = pipe.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
res = pipe(
    prompt,
    max_new_tokens=256,
)
print(res[0]["generated_text"])
