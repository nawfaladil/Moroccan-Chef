import argparse
from src.rag.main import Orchestrator

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    orchestrator = Orchestrator()
    results = orchestrator.respond(args.query_text)
    generated_response, sources = results
    print(generated_response)
    for source in sources:
        print(f"\nSource: {source}")

if __name__ == "__main__":
    main()
