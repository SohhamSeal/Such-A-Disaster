def plot_count(df: pd.core.frame.DataFrame, col_list: list, title_name: str='Train') -> None:
  '''
  Display the distribution of target value in the dataset provided
  in :
  1. pie chart
  2. bar chart
  '''
  f, ax = plt.subplots(len(col_list), 2, figsize=(12, 5))
  plt.subplots_adjust(wspace=0.3)

  for col in col_list:
    # Computing value counts for each category in the column
    s1 = df[col].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    # Colors for the outer and inner parts of the pie chart
    outer_colors = ['#FF6347', '#20B2AA']
    inner_colors = ['#FFA07A', '#40E0D0']

    # Creating outer pie chart
    ax[0].pie(
        outer_sizes, colors=outer_colors,
        labels=s1.index.tolist(),
        startangle=90, frame=True, radius=1.2,
        explode=([0.05]*(N-1) + [.2]),
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 14, 'weight': 'bold'},
        shadow=True
    )

    # Creating inner pie chart
    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=0.8, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.2]),
        pctdistance=0.8, textprops={'size': 13, 'weight': 'bold', 'color': 'black'},
        shadow=True
    )

    # Creating a white circle at the center
    center_circle = plt.Circle((0,0), .5, color='black', fc='white', linewidth=0)
    ax[0].add_artist(center_circle)

    # Barplot for the count of each category in the column
    sns.barplot(
        x=s1, y=s1.index, ax=ax[1],
        palette='coolwarm', orient='horizontal'
    )

    # Customizing the bar plot
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax[1].set_ylabel('')  # Remove y label

    # Adding count values at the end of each bar
    for i, v in enumerate(s1):
      ax[1].text(v, i+0.1, str(v), color='black', fontweight='bold', fontsize=14)

    # Adding labels and title
    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col, fontweight="bold", color='black', fontsize=14)

  # Setting a global title for all subplots
  f.suptitle(f'{title_name} Dataset Distribution of {col} : real disaster(1) or not(0)', fontsize=20, fontweight='bold', y=1.05)

  # Adjusting the spacing between the plots
  plt.tight_layout()
  plt.savefig(f'..\Images\{title_name}', bbox_inches='tight', pad_inches=0)
  plt.show()


def transform_text(text):
  '''
  ............................
  '''
  new_text = " ".join([Lemmatize.lemmatize(word) for word in word_tokenize(text) if ((word not in punc) and (word not in stop_word))])

  return new_text



def preprocess(df, is_train):
  '''
  Preprocess the data, for both train and test data, as and when accordingly sent
  Display train wordclouds only, not for test
  '''
  if is_train == True:
    plot_count(df,['target'],'Train')
  
  #dropping some unnecessary columns
  dropped_df=df.drop(['id','keyword','location'],axis=1)
  dropped_df
  
  if is_train == True:
    generate_word_cloud(dropped_df['text'],'..\Images','WordCloud before cleaning')
  
  #apply changes
  changed_df=dropped_df
  changed_df['text']=dropped_df['text'].apply(cleaner_func)
  
  if is_train == True:
    generate_word_cloud(changed_df['text'],'..\Images','WordCloud after cleaning')
    
  # #apply lemmatization and stopword removal
  # punc = list(string.punctuation)
  # stop_word = stopwords.words("english")
  # Lemmatize = WordNetLemmatizer()
  
  # lemme_df=changed_df
  # lemme_df['text']=changed_df['text'].apply(transform_text)
  # lemme_df['text']
  
  # if is_train == True:
  #   generate_word_cloud(lemme_df['text'],'..\Images','WordCloud after lemmatization and stopword removal')
  
  # # Tokenization
  # X=lemme_df['text']
  # tokenize = Tokenizer(oov_token="<OOV>")
  # tokenize.fit_on_texts(X)
  # word_index = tokenize.word_index

  # data_sequance = tokenize.texts_to_sequences(X)

  # # Padding_Sequences
  # data_padding = pad_sequences(data_sequance, maxlen=150, padding="pre", truncating="pre")

  

def take_data():
  '''
  Load data from the source data folder and do the necessary work
  '''
  train_df=pd.read_csv('..\Data\train.csv')
  test_df=pd.read_csv('..\Data\test.csv')
  #sub_df=pd.read_csv('..\Data\sample_submission.csv')
  
  train_df=preprocess(train_df,True)
  test_df=preprocess(test_df,False)
  
  
  
  
  
    




def generate_word_cloud(text, save_path, title):
  '''
  generate a wordcloud for the provided data.
  text        :list/array of strings
  save_path   :the location to store teh image in
  title       :the name of the image as well as the title  
  '''
  vectorizer=CountVectorizer()
  X=vectorizer.fit_transform(text)
  words=vectorizer.get_feature_names_out()
  word_counts=X.sum(axis=0).A1 #get the column sum of each word and then flatten it into a row array
  word_freq = dict(zip(words, word_counts))

  # generate word cloud
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

  # display wordcloud
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.title(title)
  plt.savefig(os.path.join(save_path, title), bbox_inches='tight', pad_inches=0)
  plt.show()  


def cleaner_func(text):
  '''
  apply the following transformations:
  1. to lowercase
  2. replace wrong symbols
  3. html links
  4. email ids
  5. emoticons
  6. dates
  7. usernames
  8. ip addresses
  9. symbols
  10. replace words with full forms
  11. newlines
  12. double/extra spaces
  text: string on which we need to apply the changes
  '''
  #changing all characters to lowercase
  text=str.lower(text)

  #remove some necessary text ASCII characters
  for val in replace_word_symbols:
    text=re.sub(val,'',text)

  #remove html links (also includes: http://www.amazon.co.jp/エレクトロニクス-デジタルカメラ-ポータブルオーディオ/b/ref=topnav_storetab_e?ie=UTF8&node=3210981)
  no_html=r'http[s]?://(?:[\w]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9\w][0-9\w]))+'

  #remove e-mail ids (https://stackoverflow.com/questions/2049502/what-characters-are-allowed-in-an-email-address): (also includes: mason@日本.com)
  no_emails=r"[\w0-9!#$%.(),:;<>@[\]&'\"*+-/=?^_`{|}~]+@[\w0-9_]+\.[\w0-9]+"

  #remove emoticons
  no_emoticons='[\U00010000-\U0010ffff]'

  #remove dates
  no_dates='\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](\d\d)\b'

  #removing entire profile names as otherwise the profiles will be used for classification as well!! (twitter does not want to!!)
  no_names='@[\w]+'

  #remove ip addresses
  no_ip='d{1,3}.d{1,3}.d{1,3}.d{1,3}'

  #removing all punctuation marks and symbols
  no_symbols='[^a-zA-Z0-9 \n]'

  #substitute
  lists_of_changes=[no_html, no_emails, no_emoticons, no_dates, no_names, no_ip, no_symbols]
  for val in lists_of_changes:
    text=re.sub(val,'',text)
    # print(text)

  #replacements
  replacement_changes=[replace_collocations, replace_abbreviations]
  for change_type in replacement_changes:
    for txt,re_txt in change_type.items():
      text=re.sub('\b'+txt+'\b',re_txt,text)

  #replace nextlines by a space
  text=re.sub('[\n]+',' ',text)

  #replace double or more spaces with a single space
  text=re.sub('[ ]{2,}',' ',text)

  return text