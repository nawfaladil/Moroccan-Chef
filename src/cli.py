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
    if results:
        generated_response, sources_scores = results
        print(generated_response)
        for source, score in sources_scores:
            print(f"\nSource: {source} | score={score:.4f}")
    else:
        print("no such recipe in our database.")

if __name__ == "__main__":
    main()
