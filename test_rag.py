import os
from main import ingest_documents, query_index, generate_test_documents

def test_application():
    folder_path = "test_documents"
    generate_test_documents(folder_path)  # Test documents exists
    index = ingest_documents(folder_path)  # Ingest into Pinecone

    test_cases = [
        {
            "query": "What are the warranties of Product A and B?",
            "expected_phrases": ["5-year warranty", "3-year warranty"]
        },
        {
            "query": "What features does Product A have?",
            "expected_phrases": ["15W", "white", "5-year warranty"]
        },
        {
            "query": "What are the specifications for Product B?",
            "expected_phrases": ["20W", "black"]
        },
        {
            "query": "What are the specifications for Product C?",
            "expected_phrases": ["cannot"]
        },
        {
            "query": "Which product is suitable for outdoor use?",
            "expected_phrases": ["Product A"]
        },
    ]

    for test in test_cases:
        response = query_index(index, test["query"])
        response_text = str(response)  # response is a string for comparison
        try:
            assert all(phrase.lower() in response_text.lower() for phrase in test["expected_phrases"]), (
                f"Test failed for query: {test['query']}\nExpected to find: {test['expected_phrases']}\nGot: {response_text}"
            )
            print(f"Test passed for query: {test['query']}\nResponse: {response_text}")
        except AssertionError as e:
            print(str(e))

if __name__ == "__main__":
    test_application()
