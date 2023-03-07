from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import time

model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

batch_size = 8

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English. "
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English. "
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains."
    "The unicorns spoke really great perfect American English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda:0').repeat(batch_size, 1)

print('Input size: ', input_ids.shape)

print(input_ids[0])

with torch.no_grad():
    # warmup
    for _ in range(3):
        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
        )

    # benchmark
    for _ in range(1):
        torch.cuda.synchronize()
        start_time = time.time()
        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=1024,
        )
        torch.cuda.synchronize()
        print(f"Generation took {time.time() - start_time:.2f} seconds")
print(gen_tokens.shape)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
