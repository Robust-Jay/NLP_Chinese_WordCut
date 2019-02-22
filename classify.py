import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def seg_sentence(sentence, stopwords_path):
    """
    对句子进行分词
    """
    # print "now token sentence..."

    def stopwordslist(filepath):
        """
        创建停用词list ,闭包
        """
        stopwords = [line.decode('utf-8').strip() for line in open(filepath, 'rb').readlines()]
        return stopwords

    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist(stopwords_path)  # 这里加载停用词的路径
    outstr = ''  # 返回值是字符串
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


def tokenFile(file_path, write_path):
    """
    对文本进行分词，结果存储在write_path
    :param file_path:
    :param write_path:
    :return:
    """
    with open(write_path, 'w', encoding='utf-8') as w:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                token_sen = seg_sentence(line.split('\t')[1], 'stopwords.txt')
                w.write(line.split('\t')[0] + "\t" + token_sen + "\n")


def constructDataset(path):
    """
    path: file path
    rtype: lable_list and corpus_list
    """
    label_list = []
    corpus_list = []
    with open(path, 'r') as p:
        for line in p.readlines():
            label_list.append(line.split('\t')[0])
            corpus_list.append(line.split('\t')[1])
    return label_list, corpus_list




if __name__ == '__main__':
    file_path = 'cnews/dataSet.txt'
    write_path = 'cnews/dataSet_token.txt'
    #分词
    tokenFile(file_path, write_path)

    label, data = constructDataset(write_path)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25,
                                                        random_state=42)
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)
    tfidf_vect.fit(x_train)
    xtrain_tfidf = tfidf_vect.transform(x_train)
    xtest_tfidf = tfidf_vect.transform(x_test)

    mnb_count = MultinomialNB()
    mnb_count.fit(xtrain_tfidf, y_train)
    print(mnb_count.score(xtest_tfidf, y_test))

