import os
import discord
import requests
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

keyword_dict = {"xun yu classes": "meng 420 & meng 270", "xun yu office hours": "10:00-12:00pm(mon,tues, wed & thurs)",
                "qin liu classes": "meng 201, meng 211, meng 438/602",
                "qin liu office hours": "3:30-4:30pm(mon & wed) & 3:15-5:15pm(thurs)",
                "ahmadreza baghaie classes": "eeng 860, eeng 211/212, eeng 403",
                "ahmadreza baghaie office hours": "10:00-12:00pm(mon) & 2:00-4:00pm(tues)",
                "michael colef classes": "csci 155 w01 & w02, csci 503 w01, csci 620/445 w01, itec 445 w01, csci/eeng 641 m01 w01",
                "michael colef office hours": "9:00-11:00am(tues & thurs)",
                "david nadler classes": "etec 245, bioe 620, envt 620", "david nadler office hours": "2:00-4:00pm(wed)",
                "xueqing huang classes": "csci 415/657, csci 335, csci 436/636,dtsc 701",
                "xueqing huang office hours": "3:20-5:20pm(mon,tues & thurs)",
                "steven lu classes": "aeng 490 w01, meng 349 w04,meng 507 w01",
                "steven lu office hours": "12:20-2:20pm(mon & wed)", "dorinamaria carka classes": "meng 211,221,63",
                "dorinamaria carka office hours": "11:00-12:30pm(tues & thurs) & 10:00-12:30(wed)",
                "robert n. amundsen classes": "engy 710, engy 740, ieng 285",
                "robert n. amundsen office hours": "3:30-5:30pm(mon,wed & fri)",
                "ayat jafari classes": "eeng 751 m01,w01 & bioe 751w01, eeng 382/515 w01, eeng 315 w01",
                "ayat jafari office hours": "4:00-5:30pm(tues,wed & thurs)",
                "lazaros pavlidis classes": "etec 120 w01, ctec 336 w01, ctec 336 w01l, etec 120 w01l, ctec 350 w01",
                "lazaros pavlidis office hours": "12:30-2:00pm(mon & wed) & 9:00-9:30(wed & fri)",
                "kiran balagani classes": "csci 270, csci 345, dtsc 620",
                "kiran balagani office hours": "15:00-6:00pm(mon) & 2:00-3:00pm(tues & thurs) & 1:00-2:00pm(wed)",
                "fang li classes": "csci 270, csci 345, dtsc 620", "fang li office hours": "1:00-5:00pm(fri)",
                "tao zhang classes": "csci 125 w01/w02",
                "tao zhang office hours": "3:45-4:45pm(mon & wed) & 10:15-11:15am(thurs)",
                "aydin farajidavar classes": "eeng 125/csci 135, eeng/bioe 650, eeng 270 eeng 491",
                "aydin farajidavar office hours": "10:00am-12:pm(wed) & 3:30-5:30pm(tues)",
                "tindaro loppolo classes": "meng 340, meng 604, meng 212",
                "tindaro loppolo office hours": "12:00-1:50pm(mon,tues & thurs)",
                "batu chalise classes": "eeng 770 & eeng 611",
                "batu chalise office hours": "5:00-7:00pm(mon) & 2:00-4:00 pm(thurs)",
                "sarah meyland classes": "envt 601 f01, envt 720 f01, envt 750 f01, envt 802 w01",
                "sarah meyland office hours": "5:00-6:00pm(wed);and all other days and weekly meeting by appointment",
                "maherukh akhtar classes": "csci 235 w01/w02, csci 380 w01/m01, etcs 108 w01, csci 665 w01/m01",
                "maherukh akhtar office hours": "2:00-2:40pm(mon & wed) & 11:00am-2:00pm(tues)",
                "frank lee classes": "csci 455, csci 651, csci 300",
                "frank lee office hours": "1:00-2:00pm(mon & wed) & 3:30-5:30pm(tues)",
                "sahiba wadoo classes": "eeng 715, eeng 710, eeng 320",
                "sahiba wadoo office hours": "4:00-5:30pm(wed) & 6:30-8:30pm(zoom) & 10:30am-12:00pm(fri)",
                "lak amara classes": "etec 410 m01, etec 410 m01l, ctec 241 m01, ctec 241 m01l",
                "lak amara office hours": "11:00am-12:00pm(mon) & 2:30-4:30pm(mon & wed) & 2:00-3:00pm(thurs)",
                "yoshi saito classes": "eeng 270 m01, eeng 371 m01, eeeng 403 m01",
                "yoshi saito office hours": "1:00-2:00pm(t) & 12:30-2:00pm(wed & fri)",
                "maryam ravan classes": "eeng 125/csci 135, eeng 125/csci 135, eeng 860/csci 860",
                "maryam ravan office hours": "10:00-2:00pm(mon & fri)",
                "reza amineh classes": "eeng 765, eeng 491, eeng 489",
                "reza amineh office hours": "1:00-4:00pm(mon,tues & thurs)",
                "anand santhanakrishnan classes": "eeng 125/csci 135, eeng 211, eeng 221",
                "anand santhanakrishnan office hours": "12:00-2:00pm(mon & wed)",
                "houwei cao classes": "csci 426/626 & csci 436/636,dtsc 701",
                "houwei cao office hours": "4:00-5:30pm(wed & thurs)",
                "steven billis classes": "etcs 108 m01 & etcs 108 m02",
                "steven billis office hours": "10:00-11:00am & 1:00-3pm(mon/wed)",
                "paolo gasti classes": "incs 741ma & ow incs 810 ma and ow",
                "paolo gasti office hours": "3:00-5:00pm(mon & thurs)", "n.sertac artan classes": "eeng 281 & eeng 310",
                "n.sertac artan office hours": "1:00-2:00pm(tues & thurs) & 3:00-4:00pm(fri)",
                "richard meyers classes": "ctec 204, ctec 208, ctec 243, ctec 247, etec 325",
                "richard meyers office hours": "11:10am-12:10pm(mon & wed) & 9:30-10:30am(tues & thurs) & 2:00-4:00pm(fri)",
                "jerry cheng classes": "dtsc 615 & dtsc 630", "jerry cheng office hours": "3:30-5:30pm(mon & thurs)",
                "susan gass classes": "csci 235 m01/m02, csci 235 m03/m04, csci 330/509 m01/m02, csci 330/509 m03/m04, csci/itec 620/445",
                "hi": "Hey there!", "hey": "Hey there!", "hello": "Hey there!"}


roary_positive_responses = [
    "Oh, this is very kind of you.",
    "Thanks!",
    "Thank you very much!",
    "Actually I think you are the best.",
    "Have an awesome day :)",
    "Looks like we will make good friends.",
    "Ohh that's so sweet.",
    "Nice approach well done mate."
]

roary_negative_responses = [
    "Don't be an asshole",
    "Now this is rude",
    "You can't go anywhere with that attitude",
    "You sounding like samiha right now cut it out",
    "Tough day, huh?",
    "I thought we were friends"
]


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
    if message_text in keyword_dict.keys():
        await message.channel.send(keyword_dict[message_text])

    elif message_text == 'inspire':
        quote = get_inspirational_quote()
        await message.channel.send(quote)

    else:
        message_vectorizer = TfidfVectorizer()
        message_vector = message_vectorizer.fit_transform([message_text])
        word_names = message_vectorizer.get_feature_names_out()
        dense = message_vector.todense()
        dense_list = dense.tolist()
        word_df = pd.DataFrame(dense_list, columns=word_names)
        print(word_df)
        message_text_to_array = [message_text]
        x_new_sample = vectorizer.transform(message_text_to_array)
        result = log_reg_classifier.predict(x_new_sample)[0]

        if result == 0:
            response = np.random.choice(roary_negative_responses, 1)[0]
            await message.channel.send(response)

        elif result == 1:
            response = np.random.choice(roary_positive_responses, 1)[0]
            await message.channel.send(response)

client.run(TOKEN)
