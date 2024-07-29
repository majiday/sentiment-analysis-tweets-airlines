import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_word_clouds(entity, topics):
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    for i, topic in enumerate(topics):
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(' '.join(topic))
        axs[i].imshow(wordcloud, interpolation='bilinear')
        axs[i].axis('off')
        axs[i].set_title(f'Topic {i+1}')
    plt.suptitle(entity, fontsize=16)
    plt.savefig(f'{entity}_word_clouds.png', bbox_inches='tight')
    plt.show()
