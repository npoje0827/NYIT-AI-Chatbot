import os
import discord

TOKEN = os.environ['CLIENT_TOKEN']
client = discord.Client()


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    message_text = message.content.lower()
    if message.author == client.user:
        return

    elif message_text == 'hi' or message_text == 'hey' or message_text == 'hello':
        response = 'Hey there!'
        await message.channel.send(response)

    elif message_text == 'samiha':
        response = 'What a hoe!'
        await message.channel.send(response)

client.run(TOKEN)
