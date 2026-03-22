import os
from dotenv import load_dotenv

from pinecone import Pinecone
import torch
import pandas as pd
from mistralai import Mistral
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import io
from PIL import Image
import base64
from typing import Final
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)
import nest_asyncio
import asyncio

load_dotenv("api.env")
PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_HOST = os.environ["PINECONE_INDEX_HOST"]
MISTRAL_API_KEY    = os.environ["MISTRAL_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

nest_asyncio.apply()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)
print("Loading embedding model...")
embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
embedding_model = AutoModel.from_pretrained("BAAI/llm-embedder")

print("Loading reranker model...")
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
reranker_model.eval()

print("Models loaded successfully!")
TOKEN: Final = TELEGRAM_BOT_TOKEN
BOT_USERNAME: Final = "suryaa_avivo_bot"
def RAG_QUERY(user_question):
    """Process user query through RAG pipeline and return response"""
    try:
        print(f"Processing query: {user_question}")

        def get_embedding(text):
            inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = embedding_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return cls_embedding.squeeze().tolist()

        vector = get_embedding(user_question)
        initial = index.query(
            namespace="avivo-assessment-suryaa",
            vector=vector,
            top_k=10,
            include_metadata=True,
            include_values=False
        )

        if not initial['matches']:
            return "Sorry, I couldn't find any relevant recipes in my database. Please try asking something else!"

        df = pd.DataFrame([
            {
                'id': item['id'],
                'chunk_text': item['metadata']['chunk_text'],
                'pinecone_score': item['score']
            }
            for item in initial['matches']
        ])

        print(f"Found {len(df)} initial results")

        inputs = reranker_tokenizer(
            [user_question] * len(df),
            df['chunk_text'].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        with torch.no_grad():
            logits = reranker_model(**inputs).logits.squeeze(-1)

        df['rerank_score'] = logits.numpy()
        df_rerank = df.sort_values(by='rerank_score', ascending=False).reset_index(drop=True)

        context_1 = df_rerank["chunk_text"][0] if len(df_rerank) > 0 else ""
        context_2 = df_rerank["chunk_text"][1] if len(df_rerank) > 1 else ""
        context_3 = df_rerank["chunk_text"][2] if len(df_rerank) > 2 else ""

        print("Generating response with Mistral...")
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)

        prompt = f"""You will receive a user question: "{user_question}"

You are provided with three recipe-related contexts:

Recipe Context 1: {context_1}

Recipe Context 2: {context_2}

Recipe Context 3: {context_3}

Your task:
1. Analyze the question and all three recipe contexts thoroughly.
2. Generate your response ONLY from the relevant cooking or recipe information in these contexts.
3. If the question does not relate to any of the three contexts, respond with: "NOT MATCH"
4. Stay strictly within the food, cooking, or recipe domain.
5. Provide a helpful, conversational answer.
6. At the end, include: "SOURCE: [Brief mention of recipe/context used]"

Provide a concise, helpful answer based solely on the provided contexts."""

        chat_response = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.9,
        )

        response = chat_response.choices[0].message.content
        print("Response generated successfully!")
        return response

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        print(error_msg)
        return error_msg


def RAG_IMAGE_QUERY(image_bytes, caption=""):
    """Process image query through RAG pipeline"""
    try:
        print(f"Processing image query with caption: {caption}")
        Image.open(io.BytesIO(image_bytes))  # Validate image
        client = Mistral(api_key=MISTRAL_API_KEY)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What be the Ideal Food in the Image. Example: Biryani, Sambar?... Just Provide the 1 line Answer like: Biryani thats it"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]

        chat_response = client.chat.complete(
            model="pixtral-large-latest",
            messages=messages
        )
        text_content = chat_response.choices[0].message.content
        print(text_content)

        response = RAG_QUERY(text_content)
        image_note = "  *Image Analysis:* I've analyzed your image"
        if caption:
            image_note += f" with your description: '{caption}'"

        return response + image_note

    except Exception as e:
        error_msg = f"Sorry, I couldn't process the image: {str(e)}"
        print(error_msg)
        return error_msg
async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ask command with text query"""
    query_text = ' '.join(context.args) if context.args else ""

    if not query_text:
        await update.message.reply_text(
            "Please provide a question!  "
            "Usage: `/ask How do I make pasta?`  "
            "Examples: "
            "• `/ask What ingredients are in bread?` "
            "• `/ask How to cook chicken curry?` "
            "• `/ask Tell me about baking techniques`",
            parse_mode='Markdown'
        )
        return

    print(f"/ask command: {query_text}")
    await update.message.chat.send_action(action="typing")
    response = RAG_QUERY(query_text)
    await update.message.reply_text(response)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help information"""
    help_message = (
        "*Suryaa AVIVO Recipe Bot Help*  "
        "I'm your AI-powered recipe assistant! I can help you with cooking questions and analyze food images.  "
        "*Available Commands:*  "
        " - `/ask <your question>` "
        "   Ask me anything about recipes or cooking "
        "   Example: `/ask How do I make pizza dough?`  "
        " - `/image` "
        "   Send an image with this command to analyze it "
        "   You can add a caption with your question  "
        " - `/help` "
        "   Show this help message"
        "*Tips:*"
        "• Be specific with your questions "
        "• For images, add a caption describing what you want to know "
        "• All answers come from my verified recipe database  "
        "Happy cooking!"
    )
    await update.message.reply_text(help_message, parse_mode='Markdown')


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /image command - prompts user to send image"""
    await update.message.reply_text(
        "*Image Analysis Mode*  "
        "Please send me an image of your food, ingredients, or recipe!  "
        "You can also add a caption with your question, like: "
        "• 'What recipe can I make with these ingredients?' "
        "• 'What dish is this?' "
        "• 'How do I cook this?'  "
        "Send the image now!",
        parse_mode='Markdown'
    )
    context.user_data['waiting_for_image'] = True


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process uploaded images"""
    print("Image received")
    await update.message.chat.send_action(action="typing")

    try:
        photo = update.message.photo[-1]
        caption = update.message.caption or ""
        photo_file = await photo.get_file()
        image_bytes = await photo_file.download_as_bytearray()

        print(f"Image downloaded: {len(image_bytes)} bytes, caption: '{caption}'")
        response = RAG_IMAGE_QUERY(bytes(image_bytes), caption)
        await update.message.reply_text(response, parse_mode='Markdown')
        context.user_data['waiting_for_image'] = False

    except Exception as e:
        error_msg = f"Sorry, I couldn't process your image: {str(e)}"
        print(error_msg)
        await update.message.reply_text(error_msg)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages"""
    text = update.message.text or ""
    if not text.strip():
        return

    print(f"Regular message: {text}")
    await update.message.reply_text(
        "Hi! To ask me a question, please use:  "
        "`/ask your question here`  "
        "Or use `/help` to see all available commands!",
        parse_mode='Markdown'
    )


async def error_handler(update, context):
    error_msg = str(context.error)
    print(f"Error occurred: {error_msg}")
    if "Conflict" in error_msg or "terminated by other getUpdates" in error_msg:
        print("Please stop all other running instances of this bot.")
        return
    if update and update.message:
        await update.message.reply_text("Sorry, something went wrong. Please try again!")
async def main():
    try:
        print("Initializing Telegram Bot...")
        app = Application.builder().token(TOKEN).build()

        app.add_handler(CommandHandler("ask", ask_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("image", image_command))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.Command(), handle_message))
        app.add_error_handler(error_handler)

        await app.initialize()
        await app.start()
        print(" Bot initialized successfully!")
        print("Starting polling...")

        await app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )

        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()

    except Exception as e:
        print(f" Failed to start bot: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(" Bot stopped by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
