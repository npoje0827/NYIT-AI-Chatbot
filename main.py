import os
import discord
import requests
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TOKEN = os.environ['CLIENT_TOKEN']
client = discord.Client()

# Instantiate Logistic Regression classifier for sentiment analysis
log_reg_classifier = LogisticRegression()
df = pd.read_csv('sentences_sentiment_labelled.txt', names=['sentence', 'label'], sep='\t')

# Assign x variable to include independent predictor attributes
x = df['sentence'].values

# Assign label values to y variable
y = df['label'].values

# Create variables to hold x and y train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize vectorizer and fit using x training data. Use this to vectorize x data by number of word occurrences.
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)

# Fit Logistic Regression model with vectorized x data and y data
log_reg_classifier.fit(x_train, y_train)

# Instantiate ChatBot instance from chatterbot library and configure logic adapters
roary_chatbot = ChatBot('Roary', logic_adapters=[
    {
        'import_path': 'chatterbot.logic.BestMatch',
        'default_response': 'I am sorry, but I do not understand.',
        'maximum_similarity_threshold': 0.90
    }
])

# Drop previously used SQLLite DB for the bot
roary_chatbot.storage.drop()

# Instantiate trainer instance for the ChatBot and use seven corpora to train it
trainer = ChatterBotCorpusTrainer(roary_chatbot)
trainer.train("chatterbot.corpus.english.greetings",
              "chatterbot.corpus.english.conversations",
              "chatterbot.corpus.english.botprofile",
              "chatterbot.corpus.english.health",
              "chatterbot.corpus.english.computers",
              "chatterbot.corpus.english.emotion",
              r"C:\Users\npoje\PycharmProjects\RoaryBot\NYIT.yml")


@client.event
async def on_ready():
    # Configure pandas to display all columns and rows from dataframes output to console
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Print to console once bot has connected to Discord
    print(f'{client.user} has connected to Discord!')


def get_inspirational_quote():
    # Make GET request to zenquotes API
    response = requests.get("https://zenquotes.io/api/random")
    json_data = json.loads(response.text)

    # Parse JSON returned to display quote and author
    quote = json_data[0]['q'] + " -" + json_data[0]['a']
    return quote


def analyze_sentiment(message):
    # Vectorize message by counting occurrence of words, while also removing stop words and punctuation
    message_text_to_array = [message]
    x_new_sample = vectorizer.transform(message_text_to_array)

    # Make sentiment prediction with Logistic Regression classifier
    result = log_reg_classifier.predict(x_new_sample)[0]
    sentiment_detected = "positive" if result == 1 else "negative"
    print("Sentiment detected: ", sentiment_detected)

    # Instantiate sentiment analyzer to calculate overall sentiment from compound score
    overall_sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = overall_sentiment_analyzer.polarity_scores(message)
    if sentiment_dict['compound'] >= 0.05:
        return "Positive"
    elif sentiment_dict['compound'] <= - 0.05:
        return "Negative"
    else:
        return "Neutral"


def calculate_tf_idf_scores(message):
    # Vectorize message to calculate tf idf scores for each word
    message_vectorizer = TfidfVectorizer()
    message_vector = message_vectorizer.fit_transform([message])
    word_names = message_vectorizer.get_feature_names_out()
    dense = message_vector.todense()
    dense_list = dense.tolist()

    # Create dataframe to store tf idf scores for each word and print contents
    word_df = pd.DataFrame(dense_list, columns=word_names)
    print(word_df)


@client.event
async def on_message(message):
    # Do not respond to incoming message sent from this bot
    if message.author == client.user:
        return

    # Convert incoming message to lowercase to avoid casing issues
    message_text = message.content.lower()

    # Generate inspirational quote to send
    if message_text == 'inspire':
        quote = get_inspirational_quote()
        await message.channel.send(quote)

    # Generate embedded message with guidance for commands and features bot supports
    elif message_text == '!help':
        embedVar = discord.Embed(title="Command Features", description="Here are my commands and how they work! Have "
                                                                       "fun", color=0x0000FF)
        embedVar.add_field(name="- Professor Office Hours",
                           value="Send me a message containing the professors name you would like to "
                                 "view, along with 'office hours' and receive their "
                                 "office hours!", inline=False)
        embedVar.add_field(name="- Student Resources", value="New York Tech Resources you may not have been aware of "
                                                             "to help you succeed.. whether it be to academic support "
                                                             "to the wellness center, or more..", inline=False)
        embedVar.add_field(name="- Registration Help", value="Need help with registration? Confused how to register "
                                                             "for classes and see what courses you have remaining? "
                                                             "Ask me!", inline=False)
        embedVar.add_field(name="- Course Syllabus'", value="Send me a message with the course you'd like to view and "
                                                            "'syllabus' you'd like to see! Quick and easy",
                           inline=False)
        embedVar.add_field(name="- Professor Classes", value="Send me a message containing the professors name and "
                                                             "'classes' in order to view their classes!", inline=False)
        embedVar.add_field(name="- Forms", value="Easy way to get NYIT forms. Input !forms to see what forms I provide",
                           inline=False)

    # Generate embedded message with registration help resource if requested
    elif message_text == 'registration help':
        embedVar = discord.Embed(title="NYIT Registration", color=0xff0000)
        embedVar.add_field(name="Registration Help", value="[Registration Help](https://mcusercontent.com/47ad8f893c27025c5b50a447e/files/67261c4c-a3c9-e4fd-e590-ddaa52218cf4/NYIT_Guide_How_to_Search_and_Register_for_Classes_2022_Edition.pdf)")
        await message.channel.send(embed=embedVar)

    # If specific keyword isn't detected, send message to bot to analyze and produce response
    else:
        # Call helper method to calculate tf idf scores for each word in message
        calculate_tf_idf_scores(message_text)

        # Produce response to input statement
        response = roary_chatbot.get_response(message_text)

        # Call helper method to calculate overall sentiment of input statement
        overall_sentiment = analyze_sentiment(message_text)

        # Bot is trained to send links in response to syllabus requests. Send embed with link if syllabus is requested.
        if str(response).startswith('https'):
            embedVar = discord.Embed(title="NYIT Syllabus", color=0xff0000)
            embedVar.add_field(name="Syllabus", value="[Syllabus](" + str(response) + ")")
            await message.channel.send(embed=embedVar)

        # Else send generated response with sentiment that was detected from input statement
        else:
            await message.channel.send("Sentiment detected: " + overall_sentiment + "\n" + str(response))

# Run and authenticate Discord Client using unique token
client.run(TOKEN)
