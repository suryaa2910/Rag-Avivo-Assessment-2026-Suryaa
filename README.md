# AVIVO-SURYAA-ASSESSMENT-RAG

## HOW TO RUN THE FILE
1. Install all the Dependicy from the `requirement.txt` file
2. Setup the `api.env` file with all the api's
3. Run the `Pinecone_Upserting` first (only once)
4. Run the bot:`Telegram_bot.py`
5. Open Telegram and test it
   Search for @suryaa_avivo_bot in Telegram and try:
   /ask How to make Hyderabadi Biryani?
   /help
   Send a food image

## AVIVO-RAG BASE Pipeline – Detailed Step-by-Step Explanation
This repository implements a complete Retrieval-Augmented Generation (RAG) system based on the attached Food document(s).
The pipeline consists of exactly these **3 phases** as originally designed:
### 1. Document Preparation and Embedding
**Goal**: Convert the raw Food document into searchable vector representations.

1. **Start**  
   - Source Document: The Food document(s) provided in the repository (PDF/TXT/DOCX format).

2. **Action**: Extract Text from the document  
   - All text from the entire document is extracted in one pass.

3. **Data Output**  
   - Create a CSV file with exactly these two columns (at this stage):  
     | File Name       | Extracted Text                  |  
     |-----------------|---------------------------------|  
     | food_doc.pdf    | "Full raw extracted text..."    |

4. **Decision Point (Chunking)**  
   - Evaluate total Text Length and Semantic Cohesion:  
     - **Yes, Chunk** → If the document is long (> ~1000 tokens or multiple sections), apply chunking:  
       - Chunk size: maximum 256 tokens  
       - Chunk overlap: 20 tokens  
       - Split by paragraphs/sentences using RecursiveCharacterTextSplitter (LangChain)  
     - **No, Don't Chunk** → If the document is short and self-contained, keep as single piece.  
   → Note: In the current run, chunking was skipped because the document was evaluated as short/cohesive.
***IMPORTANT WHY 256 as Max Token, 175 as Min Token and 20 as Overlap. There is a paper called : Best Practice for RAG and from the Research we get this numbers for your Reference the Link was attached here: https://arxiv.org/pdf/2407.01219***

5. **Action**: Generate LLM Dense Embedding  
   - Model used: `BAAI/llm-embedder` (HuggingFace)  
   - Each (chunk or full text) is converted into a dense vector of 768 dimensions  

6. **Data Output**  
   - The same CSV is extended with a third column:  
     | File Name       | Extracted Text                  | Embedding                          |  
     |-----------------|---------------------------------|------------------------------------|  
     | food_doc.pdf    | "Full text or chunk..."         | [0.12, -0.34, ..., 0.87] (1024-dim)|  
   - Final file: `documents_with_embeddings.csv`

## 2. Indexing and Retrieval

**Goal**: Store embeddings in a vector database and retrieve the most relevant pieces for any query.

1. **Action**: Create Index in Pinecone (Vector Database)  
   - Index name: `avivo-assessment-suryaa`  
   - Dimension: 768  
   - Metric: cosine  
   - Environment & API key loaded from environment variables

2. **Action**: Upsert (upload) the documents and their embeddings into the Pinecone Index  
   - Each row from the CSV becomes one vector in Pinecone  
   - Vector ID format: `<ured: `{filename}_chunk_{i}` (or just `{filename}` if no chunking)  
   - Metadata stored: `file_name`, `text`

3. **Input**: Receive a Test Query (any natural language question about food)

4. **Action**: Retrieve Top K=10 Documents  
   - Query text → embedded with the same `BAAI/llm-embedder` model  
   - Pinecone similarity search → returns top 10 most similar vectors

5. **Action**: Apply Reranking  
   - Model used: `BAAI/bge-reranker-v2-m3` (cross-encoder)  
   - Input: original query + the 10 retrieved texts  
   - Output: re-ordered list with accurate relevance scores  
   - Top 3 results after reranking are kept as final context

## 3. Generation

**Goal**: Produce the final accurate answer using the refined context.

1. **Input**  
   - Reranked Context (concatenated text of top reranked documents)  
   - Original Query

2. **Send to API-based Mistral LLM**  
   - Model: Mistral Large / Mixtral 8x22B via official Mistral API (or compatible endpoint)  
   - Prompt structure:  
     ```
     Use only the following context to answer the question.
     Context:
     {reranked_context}
     
     Question: {query}
     Answer:
     ```

3. **Output**  
   - The Mistral LLM generates and returns the Final Response.

4. **End**  
   - Final Response is printed/delivered to the user.

This pipeline is fully reproducible and follows exactly the original design with no additional components added.



## AVIVO-TELEGRAM BOT Pipeline.
 **AVIVO-Assessment-Suryaa Telegram Bot** should work, broken down step by step.

### Overall Purpose of the Bot
A Telegram bot that helps users with food recipes in 3 different ways:
1. Text-based query (e.g., “Biryani”)
2. Sending a photo of a dish → bot tells what it is and gives the recipe
3. Asking for help/about the bot

### Supported Commands / Flows
The bot recognizes these 3 main entry points:

| Command       | Trigger                  | What Happens |
|---------------|--------------------------|--------------|
| `/ask`        | User types `/ask Biryani` or `/ask` then types the dish name | Text-based recipe request |
| `/image`      | User types `/image` and then sends a photo | Image-based recipe identification |
| `/help`       | User types `/help`       | Bot explains itself |

You can also make it work without strict commands (more user-friendly), but for clarity we’ll explain with commands first.

### 1. `/ask` Flow (Text Query)

**User flow:**
```
/ask Chicken Biryani
```

**Bot actions:**
1. Extract the dish name (“Chicken Biryani”)
2. Send this query to your **RAG system** (Retrieval-Augmented Generation)
3. RAG searches your recipe database/knowledge base
4. Returns the full recipe (ingredients + steps)
5. Bot replies with the formatted recipe
6. At the end of the reply, bot adds the **source** (e.g., “Source: Sanjeev Kapoor Recipes”, or website URL, or your internal DB name)

**Example reply:**
```
Chicken Biryani Recipe (Hyderabadi Style)

Ingredients:
- 500g chicken
- 2 cups basmati rice
...
Steps:
1. Marinate chicken with yogurt, spices...
...
Source: Sanjeev Kapoor Recipes
```

### 2. `/help` Flow

**User types:** `/help` 

**Bot replies with something like:**
```
  "*Suryaa AVIVO Recipe Bot Help*\n\n"
        "I'm your AI-powered recipe assistant! I can help you with cooking questions and analyze food images.\n\n"
        "*Available Commands:*\n\n"
        " - `/ask <your question>`\n"
        "   Ask me anything about recipes or cooking\n"
        "   Example: `/ask How do I make pizza dough?`\n\n"
        " -  `/image`\n"
        "   Send an image with this command to analyze it\n"
        "   You can add a caption with your question\n"
        "   Example: Send a food photo with `/image What recipe is this?`\n\n"
        " -  `/help`\n"
        "   Show this help message\n\n"
        " *Tips:*\n"
        "• Be specific with your questions\n"
        "• For images, add a caption describing what you want to know\n"
        "• All answers come from my verified recipe database\n\n"
        "Happy cooking!
```

### 3. `/image` Flow (Most Advanced & Cool Feature)

**User flow:**
```
/image
→ User sends a photo of food (e.g., a plate of Dosa or Pizza)
```

**Bot actions (step by step):**

1. Bot receives the image
2. Bot sends the image to **Mistral Pixtral** vision model
3. Pixtral analyzes the image and returns a description such as:
   > "This is a South Indian Masala Dosa served with coconut chutney and sambar."
4. Bot takes Pixtral’s answer (the dish name/description) and uses it as a query
5. Sends that description to your **RAG system**
6. RAG retrieves the most relevant recipe from your database
7. Bot replies with:
   - First: What the dish is (from Pixtral)
   - Second: Full recipe (from RAG)
   - Third: Source

### Technical Flow Summary (Backend)

```
User → Telegram → Your Bot Code
       ↓
   Detect command:
   ├── /ask → extract text → send to RAG → format recipe + source → reply
   ├── /help → send static help message
   └── /image → wait for photo
            → receive photo
            → forward image to Pixtral model
            → get dish description
            → send description to RAG
            → get recipe + source
            → reply with identification + recipe + source
```
