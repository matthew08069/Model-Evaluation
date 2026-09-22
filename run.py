from mlx_lm import load, stream_generate
import json

repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(repo)

with open('testcase.jsonl', 'r', encoding='utf-8') as file:
    for index, line in enumerate(file):
        if line.strip():
            try:
                testcase = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError at line {index + 1}: {e}")
                print("\n" + "-"*50 + "\n")
                continue

            try:
                test_id = testcase['id']
                prompt = testcase['input']
            except KeyError as e:
                print(f"KeyError at line {index + 1}: {e}")
                print("\n" + "-"*50 + "\n")
                continue

            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
            )
            print(f"Test Case ID: {test_id}\n")
            for response in stream_generate(model, tokenizer, formatted_prompt, max_tokens=512):
                print(response.text, end="", flush=True)
            print("\n" + "-"*50 + "\n")