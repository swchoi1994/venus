"""Basic Venus usage example"""

from venus import LLM, SamplingParams

# Simple as that - auto-downloads model, auto-optimizes for your hardware
llm = LLM(model="demo-model")

# Single prompt
prompt = "The meaning of life is"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

outputs = llm.generate(prompt, sampling_params)
print(outputs[0].outputs[0].text)

# Batch processing
prompts = [
    "Python is a programming language that",
    "The capital of France is",
    "Artificial intelligence will"
]

outputs = llm.generate(prompts, sampling_params)
for i, output in enumerate(outputs):
    print(f"\nPrompt {i}: {prompts[i]}")
    print(f"Response: {output.outputs[0].text}")