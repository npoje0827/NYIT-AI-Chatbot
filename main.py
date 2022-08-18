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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)
log_reg_classifier.fit(x_train, y_train)

roary_chatbot = ChatBot('Roary', logic_adapters=[
    {
        'import_path': 'chatterbot.logic.BestMatch',
        'default_response': 'I am sorry, but I do not understand.',
        'maximum_similarity_threshold': 0.90
    }
]
                        )
roary_chatbot.storage.drop()

trainer = ChatterBotCorpusTrainer(roary_chatbot)
trainer.train("chatterbot.corpus.english.greetings",
              "chatterbot.corpus.english.conversations",
              "chatterbot.corpus.english.botprofile",
              "chatterbot.corpus.english.health",
              "chatterbot.corpus.english.computers",
              "chatterbot.corpus.english.emotion",
              r"C:\Users\npoje\PycharmProjects\RoaryBot\NYIT.yml")



def get_inspirational_quote():
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

    if message_text == 'inspire':
        quote = get_inspirational_quote()
        await message.channel.send(quote)

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


    elif message.content.startswith('!forms'):
        embedVar = discord.Embed(title="Forms", description="Here are the forms I support!", color=0xff0000)
        embedVar.add_field(name='Course Changes', value="Add/Drop Form,\nApplication to Change Campus,\nApplication to "
                                                        "Change Undergraduate Major,\nApplication to Declare Undergraduate "
                                                        "Minor,\nApproval to Register in a Closed Online Section,"
                                                        "\nGrade Appeals Procedure,\nPermission to Take Courses at Another "
                                                        "College,\nRequest for Challenge Examination,\nRequest to Withdraw "
                                                        "from a Course,\nRequest to Withdraw from All Courses",
                           inline=True)
        await message.channel.send(embed=embedVar)

    else:
        message_vectorizer = TfidfVectorizer()
        message_vector = message_vectorizer.fit_transform([message_text])
        word_names = message_vectorizer.get_feature_names_out()
        dense = message_vector.todense()
        dense_list = dense.tolist()
        word_df = pd.DataFrame(dense_list, columns=word_names)
        print(word_df)
        overall_sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiment_dict = overall_sentiment_analyzer.polarity_scores(message_text)
        overall_sentiment = ''
        if sentiment_dict['compound'] >= 0.05:
            overall_sentiment = "Positive"
        elif sentiment_dict['compound'] <= - 0.05:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        response = roary_chatbot.get_response(message_text)

        if str(response).startswith('https'):
            embedVar = discord.Embed(title="NYIT Syllabus", color=0xff0000)
            embedVar.add_field(name="Syllabus", value="[Syllabus](" + str(response) + ")")
            await message.channel.send(embed=embedVar)
        else:
            await message.channel.send("Sentiment detected: " + overall_sentiment + "\n" + str(response))
        message_text_to_array = [message_text]
        x_new_sample = vectorizer.transform(message_text_to_array)
        result = log_reg_classifier.predict(x_new_sample)[0]
        sentiment_detected = "positive" if result == 1 else "negative"
        print("Sentiment detected: ", sentiment_detected)


client.run(TOKEN)
