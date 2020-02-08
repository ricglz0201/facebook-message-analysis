import analyzer
import numpy as np
import matplotlib.pyplot as plt

# Generate subplots
fig, ax_array = plt.subplots(2, 3)

def show_daily_total_graph(ax, xdata, ydata, ydata_stickers):
    indices = np.arange(len(xdata))

    ax.plot(indices, ydata,
            alpha=1.0, color='dodgerblue',
            label='All messages')

    ax.plot(indices, ydata_stickers,
            alpha=1.0, color='orange',
            label='Facebook stickers')

    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title('Number of messages exchanged every day')

    num_ticks = 16 if len(indices) >= 16 else len(indices)
    tick_spacing = round(len(indices) / num_ticks)
    ticks = [tick_spacing * i for i in range(num_ticks) if tick_spacing * i < len(xdata)]
    tick_labels = [xdata[tick] for tick in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    ax.legend()

def show_monthly_total_graph(ax, xdata, ydata, ydata_stickers):
    indices = np.arange(len(xdata))

    ax.bar(indices, ydata,
            alpha=1.0, color='dodgerblue',
            label='All messages')

    ax.bar(indices, ydata_stickers,
            alpha=1.0, color='orange',
            label='Facebook stickers')

    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title('Number of messages exchanged every month')

    ax.set_xticks(indices)
    ax.set_xticklabels(xdata)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    ax.legend()

def show_day_name_average_graph(ax, xdata, ydata):
    indices = np.arange(len(xdata))
    bar_width = 0.6

    ax.bar(indices, ydata, bar_width,
            alpha=1.0, color='dodgerblue',
            align='center',
            label='All messages')

    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Count')
    ax.set_title('Average number of messages every day of the week')

    ax.set_xticks(indices)
    ax.set_xticklabels(xdata)

def show_hourly_average_graph(ax, xdata, ydata):
    indices = np.arange(len(xdata))
    bar_width = 0.8

    ax.bar(indices, ydata, bar_width,
            alpha=1.0, color='dodgerblue',
            align='center',
            label='All messages')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Count')
    ax.set_title('Average number of messages every hour of the day')

    ax.set_xticks(indices)
    ax.set_xticklabels(xdata)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

def show_daily_sentiment_graph(ax, xdata, ydata):
    indices = np.arange(len(xdata))

    ax.plot(indices, ydata,
            alpha=1.0, color='darkseagreen',
            label='VADER sentiment')

    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment')
    ax.set_title('Average sentiment over time')

    num_ticks = 16 if len(indices) >= 16 else len(indices)
    tick_spacing = round(len(indices) / num_ticks)
    ticks = [tick_spacing * i for i in range(num_ticks) if tick_spacing * i < len(xdata)]
    tick_labels = [xdata[tick] for tick in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_ylim([-1.0, 1.0])

    ax.legend()

def show_top_words_graph(ax, xdata, ydata):
    indices = np.arange(len(xdata))
    bar_width = 0.8

    ax.barh(indices, ydata, bar_width,
            alpha=1.0, color='orchid',
            align='center',
            label='All messages')

    ax.set_ylabel('Word')
    ax.set_xlabel('Uses')
    ax.set_title('Our {0} most used words'.format(len(xdata)))

    ax.set_yticks(indices)
    ax.set_yticklabels(xdata)

def plot():
    print('Displaying ...')
    print(len(analyzer.xdata_daily))

    # Call the graphing methods
    show_daily_total_graph(ax_array[0][0], analyzer.xdata_daily, analyzer.ydata_daily, analyzer.ydata_daily_stickers)
    show_monthly_total_graph(ax_array[0][1], analyzer.xdata_monthly, analyzer.ydata_monthly, analyzer.ydata_monthly_stickers)
    show_daily_sentiment_graph(ax_array[0][2], analyzer.xdata_sentiment, analyzer.ydata_sentiment)
    show_day_name_average_graph(ax_array[1][0], analyzer.xdata_day_name, analyzer.ydata_day_name)
    show_hourly_average_graph(ax_array[1][1], analyzer.xdata_hourly, analyzer.ydata_hourly)
    show_top_words_graph(ax_array[1][2], analyzer.xdata_top_words[::-1], analyzer.ydata_top_words[::-1])

    # Display the plots
    plt.show()

    print('Done.')
