"""RAG evaluation minimal invocation example."""

from app.rag_eval import RagEvalTester


if __name__ == "__main__":
    tester = RagEvalTester()
    tester.run_insert_test_data()
    tester.run_eval()
