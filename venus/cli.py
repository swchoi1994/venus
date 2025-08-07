#!/usr/bin/env python3
"""Venus CLI tool"""

import argparse
import sys
from venus import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Venus Inference CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Inference command
    infer = subparsers.add_parser("infer", help="Run inference")
    infer.add_argument("--model", required=True, help="Model name or path")
    infer.add_argument("--prompt", required=True, help="Input prompt")
    infer.add_argument("--max-tokens", type=int, default=100)
    infer.add_argument("--temperature", type=float, default=0.7)
    infer.add_argument("--top-p", type=float, default=0.9)
    
    # Chat command
    chat = subparsers.add_parser("chat", help="Interactive chat")
    chat.add_argument("--model", required=True, help="Model name or path")
    
    # Server command
    server = subparsers.add_parser("serve", help="Start API server")
    server.add_argument("--model", required=True, help="Model name or path")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    if args.command == "infer":
        llm = LLM(model=args.model)
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        outputs = llm.generate(args.prompt, sampling_params)
        print(outputs[0].outputs[0].text)
        
    elif args.command == "chat":
        llm = LLM(model=args.model)
        print("💬 Venus Chat (type 'exit' to quit)")
        while True:
            try:
                prompt = input("\nYou: ")
                if prompt.lower() in ["exit", "quit"]:
                    break
                
                messages = [{"role": "user", "content": prompt}]
                response = llm.chat(messages)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                break
        
    elif args.command == "serve":
        from venus.server import run
        run(args.model, args.host, args.port)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()