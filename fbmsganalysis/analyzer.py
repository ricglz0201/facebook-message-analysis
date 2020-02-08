from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import namedtuple, defaultdict
from operator import itemgetter
from nltk.corpus import stopwords
from unidecode import unidecode

from datetime import datetime
from heapq import nlargest
from string import punctuation
from time import clock
from json import load
from copy import deepcopy

from grapher import plot

stop_words = set(stopwords.words('spanish'))
sentiment_analyzer = SentimentIntensityAnalyzer()
cache = {}

# Empty public variables
xdata_daily = []
ydata_daily = []
ydata_daily_stickers = []
xdata_monthly = []
ydata_monthly = []
ydata_monthly_stickers = []
xdata_day_name = []
ydata_day_name = []
xdata_hourly = []
ydata_hourly = []
xdata_sentiment = []
ydata_sentiment = []
xdata_top_words, ydata_top_words = [None, None]

def _load_messages(filename):
    if filename in cache:
        return cache[filename]
    else:
        with open(filename) as jsonfile:
            data = load(jsonfile)
            cache[filename] = data
            return data

def get_messages(filename, copy_from_cache=True):
    data = _load_messages(filename)

    # Copy the stored messages we have
    copied_messages = data['messages']
    if copy_from_cache:
        copied_messages = deepcopy(data['messages'])

    # Return a sorted list of messages by time
    return sorted(copied_messages, key=lambda message : message['timestamp_ms'])

def analyze(filename):
    '''
    MESSAGE ANALYSIS
    '''
    # Global variables
    global xdata_daily, ydata_daily, ydata_daily_stickers, xdata_monthly, ydata_monthly, ydata_monthly_stickers, xdata_day_name, ydata_day_name, xdata_hourly, ydata_hourly, xdata_sentiment, ydata_sentiment, xdata_top_words, ydata_top_words

    # Load messages
    print('Reading file {0} ...'.format(filename))
    timestamp = clock()
    messages = get_messages(filename, copy_from_cache=False)
    print('Loaded {0} messages in {1:.2f} seconds.'.format(len(messages), clock() - timestamp))

    print('Aggregating data ...')
    timestamp = clock()

    # Data structures to hold information about the messages
    daily_counts = defaultdict(int)
    daily_sticker_counts = defaultdict(int)
    daily_sentiments = defaultdict(float)
    monthly_counts = defaultdict(int)
    monthly_sticker_counts = defaultdict(int)
    hourly_counts = defaultdict(int)
    day_name_counts = defaultdict(int)
    word_frequencies = defaultdict(int)
    first_date = None
    last_date = None
    content = None

    # Extract information from the messages
    for message in messages:
        # Convert message's Unix timestamp to local datetime
        ts = message['timestamp_ms'] / 1000
        date = datetime.fromtimestamp(ts)
        month = date.strftime('%Y-%m')
        day = date.strftime('%Y-%m-%d')
        day_name = date.strftime('%A')
        hour = date.time().hour

        # Increment message counts
        hourly_counts[hour] += 1
        day_name_counts[day_name] += 1
        daily_counts[day] += 1
        monthly_counts[month] += 1

        # Get content in message if it has any
        if 'content' in message:
            content = unidecode(message['content'])

            # Rudimentary sentiment analysis using VADER
            sentiments = sentiment_analyzer.polarity_scores(content)
            daily_sentiments[day] += sentiments['pos'] - sentiments['neg']
        # Increase sticker count if any stickers were added
        elif 'sticker' in message:
            daily_sticker_counts[day] += 1
            monthly_sticker_counts[month] += 1

        # Determine word frequencies
        if content:
            # Split message up by spaces to get individual words
            for word in content.split(' '):
                # Make the word lowercase and strip it of punctuation
                new_word = word.lower().strip(punctuation)

                # Word might have been entirely punctuation; don't strip it
                if not new_word:
                    new_word = word.lower()

                # Ignore word if it in the stopword set or if it is less than 2 characters
                if len(new_word) > 1 and new_word not in stop_words:
                    word_frequencies[new_word] += 1

        # Determine start and last dates of messages
        if (first_date and first_date > date) or not first_date:
            first_date = date
        if (last_date and last_date < date) or not last_date:
            last_date = date

    # Take the average of the sentiment amassed for each day
    for day, message_count in daily_counts.items():
        daily_sentiments[day] /= message_count

    # Get the number of days the messages span over
    num_days = (last_date - first_date).days

    # Get most common words
    top_words = nlargest(42, word_frequencies.items(), key=itemgetter(1))

    print('Processed data in {0:.2f} seconds.'.format(clock() - timestamp))

    print('Preparing data for display ...')

    # Format data for graphing
    xdata_daily = sorted(list(daily_counts.keys()))
    ydata_daily = [daily_counts[x] for x in xdata_daily]
    ydata_daily_stickers = [daily_sticker_counts[x] for x in xdata_daily]
    xdata_monthly = sorted(list(monthly_counts.keys()))
    ydata_monthly = [monthly_counts[x] for x in xdata_monthly]
    ydata_monthly_stickers = [monthly_sticker_counts[x] for x in xdata_monthly]
    xdata_day_name = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    ydata_day_name = [float(day_name_counts[x]) / num_days * 7 for x in xdata_day_name]
    xdata_hourly = ['{0}:00'.format(i) for i in range(24)]
    ydata_hourly = [float(hourly_counts[x]) / num_days for x in range(24)]
    xdata_sentiment = sorted(list(daily_sentiments.keys()))
    ydata_sentiment = [daily_sentiments[x] for x in xdata_sentiment]
    xdata_top_words, ydata_top_words = zip(*top_words)

    plot()
