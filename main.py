import os

import aiohttp
import discord
import requests
import json
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN = os.environ['CLIENT_TOKEN']
client = discord.Client()


def get_quote():
    response = requests.get("https://zenquotes.io/api/random")
    json_data = json.loads(response.text)
    quote = json_data[0]['q'] + " -" + json_data[0]['a']
    return quote


@client.event
async def on_ready():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    message_text = message.content.lower()
    message_vectorizer = TfidfVectorizer()
    message_vector = message_vectorizer.fit_transform([message_text])
    word_names = message_vectorizer.get_feature_names_out()
    dense = message_vector.todense()
    dense_list = dense.tolist()
    word_df = pd.DataFrame(dense_list, columns=word_names)
    print(word_df)

    if message_text == 'hi' or message_text == 'hey' or message_text == 'hello':
        response = 'Hey there!'
        await message.channel.send(response)

    elif message_text.startswith('inspire'):
        quote = get_quote()
        await message.channel.send(quote)

client.run(TOKEN)
