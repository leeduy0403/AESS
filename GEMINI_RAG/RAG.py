import re
import time
import numpy as np
import pandas as pd
import faiss
import google.generativeai as genai
from docx import Document
from load_creds import load_creds

# Configure Generative AI & Embedding Model
genai.configure(credentials=load_creds())

EMBEDDING_MODEL = "models/embedding-001"
GENERATION_MODEL = "tunedModels/asapset1model-nsaayaipvn58"

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(model_name=GENERATION_MODEL, generation_config=generation_config)


# ---------------------- STEP 1: EXTRACT & CHUNK TEXT ---------------------- #
def extract_docx_text(file_path):
    """Extract text from DOCX file."""
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting DOCX: {e}"

def chunk_text(text, chunk_size=150, overlap=30):
    """Split text into overlapping chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ---------------------- STEP 2: CREATE VECTOR DATABASE ---------------------- #
def get_embedding(text):
    """Generate embedding for a given text."""
    response = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
    return np.array(response["embedding"], dtype=np.float32)

def create_vector_database(doc_texts):
    """Convert text chunks into embeddings and store them in FAISS vector DB."""
    
    chunk_mapping = {}  
    embeddings = []  # List to store embeddings for FAISS
    idx = 0

    for doc_name, text in doc_texts.items():
        chunks = chunk_text(text)  # Split text into chunks
        for chunk in chunks:
            embedding = get_embedding(chunk)  # Get embedding
            embeddings.append(embedding)  # Store embedding
            chunk_mapping[idx] = chunk  # Map index to text chunk
            idx += 1

    # Ensure embeddings are not empty
    if not embeddings:
        raise ValueError("No embeddings were generated. Check input texts or embedding function.")

    # Convert embeddings list to NumPy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Dynamically detect embedding size
    vector_size = embeddings_array.shape[1]
    # print(f"Detected embedding size: {vector_size}")

    # Initialize FAISS index
    index = faiss.IndexFlatL2(vector_size)
    
    # Add all embeddings to FAISS index
    index.add(embeddings_array)
    
    # print(f"FAISS index built successfully with {index.ntotal} vectors.")
    # print(f"Type of index: {type(index)}")
    return index, chunk_mapping  # Return FAISS index and mapping



# ---------------------- STEP 3: RETRIEVE RELEVANT CHUNKS ---------------------- #
def retrieve_relevant_chunks(essay_text, index, chunk_mapping, top_k=3):
    """Retrieve the most relevant chunks for a given essay."""
    essay_embedding = get_embedding(essay_text)

    # Ensure the index has vectors before searching
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty. Ensure vectors were added successfully.")

    distances, indices = index.search(np.array([essay_embedding]), top_k)

    # Filter out invalid (-1) indices
    valid_indices = [i for i in indices[0] if i != -1]

    if not valid_indices:
        raise ValueError("No valid results found in FAISS search.")

    return "\n".join([chunk_mapping[i] for i in valid_indices])



# ---------------------- STEP 4: EVALUATE ESSAYS ---------------------- #
def clean_feedback(feedback):
    """Remove unwanted numbers from feedback text."""
    feedback = re.sub(r"(?<!\s)\d+", "", feedback)
    return feedback.strip()

def extract_scores_and_feedback(response_text):
    """Extract scores and feedback from AI response."""
    coherence, lexical, grammar = 0, 0, 0
    feedback = ""

    lines = response_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("coherence and cohesion:"):
            match = re.search(r"\d+", line)
            if match: coherence = int(match.group(0))
        elif line.lower().startswith("lexical resource:"):
            match = re.search(r"\d+", line)
            if match: lexical = int(match.group(0))
        elif line.lower().startswith("grammatical range and accuracy:"):
            match = re.search(r"\d+", line)
            if match: grammar = int(match.group(0))
        elif line.lower().startswith("feedback:"):
            feedback = line[len("Feedback:"):].strip()

    feedback = clean_feedback(feedback)
    ovr = min(6, max(1, coherence + lexical + grammar))
    return ovr, [coherence, lexical, grammar], feedback

def evaluate_essays(description_path, content_path, output_path):
    """Main function to evaluate essays with RAG-based retrieval."""
    # Load and chunk description & rubric
    description_text = extract_docx_text(description_path)
    
    # Create vector database
    indexing, chunk_mapping = create_vector_database({"description": description_text})

    # Load essay content
    df = pd.read_excel(content_path)

    # request_count = 0
    results = []

    for index, row in df.iterrows():
        # if request_count >= 4:  # Rate limit: 4 requests per minute
        #     time.sleep(60)
        #     request_count = 0
        
        essay_id = row["essay_id"]
        essay_content = row["essay"]

        # Retrieve relevant chunks
        retrieved_context = retrieve_relevant_chunks(essay_content, indexing, chunk_mapping)

        try:
            prompt = (
            f"DESCRIPTION: {retrieved_context}\n"
            f"CONTENT: {essay_content}\n\n"
            f"PROMPT: Evaluate the given CONTENT based on the DESCRIPTION. "
            f"Follow the exact format below in your response:\n\n"
            f"Score: (Ensure the score aligns with the range specified in the DESCRIPTION)\n"
            f"Coherence and Cohesion:\n"
            f"Lexical Resource:\n"
            f"Grammatical Range and Accuracy:\n"
            f"Feedback: (Provide clear and concise feedback without including any numbers or scores.)"
        )

            response = model.generate_content(prompt)
            response_text = response.text.strip()

            score, scores, feedback = extract_scores_and_feedback(response_text)

            results.append({
                "essay_id": essay_id,
                "ovr": score,
                "scores": scores,
                "components": [
                    "Coherence and Cohesion",
                    "Lexical Resource",
                    "Grammatical Range and Accuracy"
                ],
                "feedback": feedback
            })
            # request_count += 1

        except Exception as e:
            print(f"Error processing essay_id {essay_id}: {e}")
            results.append({
                "essay_id": essay_id,
                "ovr": 0,
                "scores": [0, 0, 0],
                "components": [
                    "Coherence and Cohesion",
                    "Lexical Resource",
                    "Grammatical Range and Accuracy"
                ],
                "feedback": "Error processing submission."
            })

    # Save results
    results_df = pd.DataFrame(results)
    final_df = df.merge(results_df, on="essay_id", how="left")
    final_df.to_excel(output_path, index=False)
    
    print(f"Evaluation complete. Results saved to {output_path}")


# ---------------------- RUN PROGRAM ---------------------- #
if __name__ == "__main__":
    evaluate_essays("C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET3//essay_set_3_description.docx", "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET3//essay_set_3.xlsx", "C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//OUTPUT//zero_shot//output_RAG_set3.xlsx")
