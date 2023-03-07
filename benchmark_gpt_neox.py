from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half().cuda()

tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

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
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English. "
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English. "
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains."
    "The unicorns spoke really great perfect American English."
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English. "
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English. "
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains."
    "The unicorns spoke really great perfect American English."
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

confs = [
    (128, 2, 1),
    (128, 32, 1),
    (128, 64, 1),
    (128, 96, 1),
    (128, 128, 1),
    # (512, 128, 1)
]

for (input_length, output_length, batch_size) in confs:
    print(f'Input: {input_length}, Output: {output_length}, Batch: {batch_size}')

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda:0').repeat(batch_size, 1)[:, :input_length]

    print('Input size: ', input_ids.shape)

    print(input_ids[0])
    print(tokenizer.batch_decode(input_ids)[0])

    with torch.inference_mode():
        # warmup
        for _ in range(3):
            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                # max_length=output_length,
                max_new_tokens = output_length,
            )

        # benchmark
        for _ in range(1):
            torch.cuda.synchronize()
            start_time = time.time()
            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                # max_length=output_length,
                max_new_tokens = output_length,
            )
            torch.cuda.synchronize()
            print(f"Generation took {time.time() - start_time:.2f} seconds")
        if output_length == 2:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                gen_tokens = model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=0.9,
                    # max_length=output_length,
                    max_new_tokens = output_length,
                )
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            prof.export_chrome_trace(f"trace_{input_length}_{output_length}_{batch_size}.json")
    print(gen_tokens.shape)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    # print(prompt)
    print(gen_text)
