import os
from dotenv import load_dotenv
import sys
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score

from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from conversational_rag_chain import conversational_rag_chain

load_dotenv()

# --- Configurable paths and model names ---
FOLDER_PATH = os.getenv("FOLDER_PATH", "data/ancient_greece")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
# You must import or define conversational_rag_chain before this script runs!

def evaluate_rag_chain(eval_json_path="evaluation/evaluation_questions.json", output_csv="evaluation/result/chatbot_evaluation_results.csv"):
    # Load evaluation data
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(eval_json_path, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    rouge = Rouge()
    results = []

    llm = ChatOpenAI(model="gpt-4o-mini")  # or gpt-4-turbo, etc.

    custom_criteria = {
        "faithfulness": "Does the answer accurately reflect the information present in the provided context?",
        "context_faithfulness": "Is the answer faithful to the retrieved context and does not introduce information not present in the context?",
    }
    evaluators = {
        "faithfulness": load_evaluator("criteria", llm=llm, criteria={"faithfulness": custom_criteria["faithfulness"]}),
        "context_faithfulness": load_evaluator("criteria", llm=llm, criteria={"context_faithfulness": custom_criteria["context_faithfulness"]}),
        "relevance": load_evaluator("criteria", llm=llm, criteria="relevance"),
        "truthfulness": load_evaluator("qa", llm=llm),
    }

    for idx, item in enumerate(evaluation_data):
        query = item["question"]
        expected_answer = item["expected_answer"]
        ground_truth_chunk = item.get("ground_truth_chunk", "")

        try:
            response = conversational_rag_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": f"eval_{idx}"}},
            )
            rag_answer = response["answer"]
            retrieved_docs = response.get("context", [])
            retrieved_texts = [doc.page_content for doc in retrieved_docs]
            retrieved_context = "\n".join(retrieved_texts)
        except Exception as e:
            rag_answer = f"Error: {e}"
            retrieved_texts = []
            retrieved_context = ""

        # Classic metrics
        try:
            vectorizer = TfidfVectorizer().fit([expected_answer, rag_answer])
            tfidf_matrix = vectorizer.transform([expected_answer, rag_answer])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception:
            cosine_sim = None

        try:
            bleu = sentence_bleu([expected_answer.split()], rag_answer.split())
        except Exception:
            bleu = None

        try:
            rouge_scores = rouge.get_scores(rag_answer, expected_answer)[0]
            rouge_1 = rouge_scores["rouge-1"]["f"]
            rouge_2 = rouge_scores["rouge-2"]["f"]
            rouge_l = rouge_scores["rouge-l"]["f"]
        except Exception:
            rouge_1 = rouge_2 = rouge_l = None

        try:
            meteor = meteor_score([expected_answer.split()], rag_answer.split())
        except Exception:
            meteor = None

        try:
            P, R, F1 = bert_score([rag_answer], [expected_answer], lang="en", rescale_with_baseline=True)
            bertscore_f1 = F1[0].item()
        except Exception:
            bertscore_f1 = None

        # LLM-based metrics
        llm_metrics = {}
        for name, evaluator in evaluators.items():
            try:
                if name == "truthfulness":
                    # For "qa" evaluator, reference is the expected answer
                    eval_result = evaluator.evaluate_strings(
                        prediction=rag_answer,
                        reference=expected_answer,
                        input=query,
                    )
                else:
                    eval_result = evaluator.evaluate_strings(
                        prediction=rag_answer,
                        reference=expected_answer,
                        input=query,
                        context=retrieved_context,
                    )
                llm_metrics[name] = eval_result.get("score", eval_result.get("value", None))
            except Exception as e:
                llm_metrics[name] = f"Error: {e}"

        results.append({
            "question": query,
            "expected_answer": expected_answer,
            "ground_truth_chunk": ground_truth_chunk,
            "rag_answer": rag_answer,
            "cosine_similarity": cosine_sim,
            "bleu_score": bleu,
            "rouge-1": rouge_1,
            "rouge-2": rouge_2,
            "rouge-l": rouge_l,
            "meteor": meteor,
            "bertscore_f1": bertscore_f1,
            **llm_metrics,
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Evaluation complete. Results saved to {output_csv}")
    print(df.head())

if __name__ == "__main__":
    evaluate_rag_chain()