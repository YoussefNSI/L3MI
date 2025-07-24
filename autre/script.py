import discord
from discord.ext import commands
import openai
import os
from dotenv import load_dotenv
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

load_dotenv('keys.env')

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Configuration du client Groq
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Stockage des conversations
conversations = defaultdict(list)
MAX_HISTORY = 8  # Nombre maximum de messages conservés
INACTIVITY_RESET = timedelta(minutes=30)  # Réinitialisation après 30 minutes d'inactivité

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)


def update_history(user_id, role, content):
    # Ajoute un message à l'historique et le nettoie
    timestamp = datetime.now()
    conversations[user_id].append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })

    # Nettoyage de l'historique
    conversations[user_id] = [
                                 msg for msg in conversations[user_id]
                                 if timestamp - msg["timestamp"] < INACTIVITY_RESET
                             ][-MAX_HISTORY:]


@bot.command(name='chat', help='Chat with context-aware AI')
async def chat(ctx, *, prompt: str):
    user_id = ctx.author.id

    try:
        # Mise à jour de l'historique avec le prompt utilisateur
        update_history(user_id, "user", prompt)

        # Préparation des messages pour l'API
        messages = [
            {"role": "system", "content": "Conversation contextuelle. Sois concis (max 1500 caractères)."},
            *[{"role": msg["role"], "content": msg["content"]}
              for msg in conversations[user_id]]
        ]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        # Mise à jour avec la réponse de l'IA
        update_history(user_id, "assistant", answer)

        # Découpage et envoi de la réponse
        chunks = [answer[i:i + 2000] for i in range(0, len(answer), 2000)]
        for i, chunk in enumerate(chunks):
            await ctx.send(f"**Réponse** ({i + 1}/{len(chunks)}):\n{chunk}")
            if i < len(chunks) - 1:
                await asyncio.sleep(1)

    except Exception as e:
        await ctx.send(f"Erreur: {str(e)}")


@bot.command(name='reset', help='Réinitialise la conversation')
async def reset_chat(ctx):
    user_id = ctx.author.id
    conversations[user_id].clear()
    await ctx.send("Conversation réinitialisée !")

@bot.command(name='tana')
async def soso(ctx):
    await ctx.send("jsuis une tana jsuis une tana donc bien évidemment queeee jmet un haut zara enft ")


@bot.command(name='resume', help='Résume les N derniers messages du salon')
async def resume(ctx, limit: int = 100):
    try:
        # Vérification des limites de sécurité
        if limit > 1000:
            return await ctx.send("⚠️ Limite maximale : 1000 messages")

        await ctx.send(f"📚 Collecte des {limit} derniers messages...")

        # Récupération de l'historique
        messages = []
        async for msg in ctx.channel.history(limit=limit):
            content = f"{msg.author.display_name} ({msg.created_at.strftime('%Y-%m-%d %H:%M')}): "
            content += msg.clean_content

            # Ajout des embeds et attachments
            for embed in msg.embeds:
                content += f"\n[EMBED: {embed.title}] {embed.description}"
            for attachment in msg.attachments:
                content += f"\n[FICHIER: {attachment.filename}] {attachment.url}"

            messages.append(content)

        # Formatage pour l'IA
        context = "\n\n".join(messages[::-1])  # Inversion pour ordre chronologique
        context = context[:15000]  # Limite de contexte pour l'API

        # Génération du résumé
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": "Tu es un assistant qui résume de longs historiques de chat. Fais un résumé structuré en français avec les points clés. Sois concis et objectif."},
                {"role": "user", "content": f"Résume cet historique de discussion :\n\n{context}"}
            ],
            max_tokens=2000,
            temperature=0.7
        )

        # Traitement de la réponse
        summary = response.choices[0].message.content

        # Découpage en chunks
        chunks = [summary[i:i + 1900] for i in range(0, len(summary), 1900)]

        await ctx.send(f"📃 **Résumé des {limit} derniers messages**")
        for i, chunk in enumerate(chunks, 1):
            try:
                await ctx.send(f"**Partie {i}:**\n{chunk}")
                if i < len(chunks):  # Attendre uniquement s'il reste des chunks
                    await asyncio.sleep(1)
            except Exception as e:
                await ctx.send(f"Erreur lors de l'envoi de la partie {i}: {e}")


    except Exception as e:
        await ctx.send(f"❌ Erreur : {str(e)}")

@bot.event
async def on_ready():
    print(f'Connecté en tant que {bot.user.name}')


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)