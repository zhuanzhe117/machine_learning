#encoding=utf-8

import sys
import os
import jieba

reload(sys)
sys.setdefaultencoding('utf-8')

def savefile(savepath,content):
    fp = open(savepath,"wb")
    fp.write(content)
    fp.close()

def readfile(path):
    fp = open(path,"rb")
    content = fp.read()
    fp.close()
    return content
#分词
def participle():
    corpus_path = "D:/materials/dataset/train_corpus_small/"
    seg_path = "D:/materials/dataset/train_corpus_seg/"
    catelist = os.listdir(corpus_path)
    # 获取每个目录下的所有文件
    for mydir in catelist:
        class_path = corpus_path + mydir + "/"
        seg_dir = seg_path + mydir + "/"
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path + file_path
            content = readfile(fullname).strip()
            content_seg = jieba.cut(content)
            savefile(seg_dir + file_path, " ".join(content_seg))
    print "中文语料分词结束！！!"

#将分词结果保存到Bunch数据结构
import pickle
from sklearn.datasets.base import Bunch
def persistence_to_Bunch():
    bunch = Bunch(target_name=[],label=[],filenames=[],contents=[])
    wordbag_path = "D:/materials/dataset/train_word_bag/train_set.dat"
    seg_path = "D:/materials/dataset/train_corpus_seg/"

    catelist = os.listdir(seg_path)
    bunch.target_name.extend(catelist)
    for mydir in catelist:
        class_path = seg_path + mydir + "/"
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path + file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname).strip())
        print mydir + "类别结构化完成！！！"
    #Bunch对象持久化
    file_obj = open(wordbag_path,"wb")
    pickle.dump(bunch,file_obj)
    file_obj.close()
    print "构建文本对象Bunch结束！！！"

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  #TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer   #TF-IDF向量生成类
#读取Bunch对象
def readbunchobj(path):
    file_obj = open(path,"rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch
#写入Bunch对象
def writebunchobj(path,bunchobj):
    file_obj = open(path,"wb")
    pickle.dump(bunchobj,file_obj)
    file_obj.close()

def get_tfidf():
    path = "D:/materials/dataset/train_word_bag/train_set.dat"
    bunch = readbunchobj(path)
    #构建TF-IDF词向量空间对象
    tfidfspace = Bunch(target_name=bunch.target_name,label = bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})
    #读取停用词表
    stopword_path = "D:/materials/dataset/ChineseStopWords.txt"
    stpwrdlst = readfile(stopword_path).splitlines()
    #使用TfidfVectorizer初始化向量空间模型
    vectorizer = TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5)
    transformer = TfidfTransformer()
    #文本转为词频矩阵，单独保存字典文件
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_
    #持久化TF-IDF向量词袋
    space_path = "D:/materials/dataset/train_word_bag/tfidfspace.dat"
    writebunchobj(space_path,tfidfspace)

########################################测试集处理################################
#分词
def participle_test():
    corpus_path = "D:/materials/dataset/test_corpus_small/"
    seg_path = "D:/materials/dataset/test_corpus_seg/"
    catelist = os.listdir(corpus_path)
    # 获取每个目录下的所有文件
    for mydir in catelist:
        class_path = corpus_path + mydir + "/"
        seg_dir = seg_path + mydir + "/"
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path + file_path
            content = readfile(fullname).strip()
            content_seg = jieba.cut(content)
            savefile(seg_dir + file_path, " ".join(content_seg))
    print "测试集中文语料分词结束！！!"

#将分词结果保存到Bunch数据结构
def persistence_to_Bunch_test():
    bunch = Bunch(target_name=[],label=[],filenames=[],contents=[])
    wordbag_path = "D:/materials/dataset/test_word_bag/test_set.dat"
    seg_path = "D:/materials/dataset/test_corpus_seg/"
    catelist = os.listdir(seg_path)
    bunch.target_name.extend(catelist)
    for mydir in catelist:
        class_path = seg_path + mydir + "/"
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path + file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname).strip())
        print mydir + "类别结构化完成！！！"
    #Bunch对象持久化
    file_obj = open(wordbag_path,"wb")
    pickle.dump(bunch,file_obj)
    file_obj.close()
    print "测试集构建文本对象Bunch结束！！！"

def get_tfidf_test():
    path = "D:/materials/dataset/test_word_bag/test_set.dat"
    bunch = readbunchobj(path)
    #构建TF-IDF词向量空间对象
    testspace = Bunch(target_name=bunch.target_name,label = bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})
    #导入训练集的词袋
    trainbunch = readbunchobj("D:/materials/dataset/train_word_bag/tfidfspace.dat")
    #读取停用词表
    stopword_path = "D:/materials/dataset/ChineseStopWords.txt"
    stpwrdlst = readfile(stopword_path).splitlines()
    #使用TfidfVectorizer初始化向量空间模型
    vectorizer = TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5,vocabulary = trainbunch.vocabulary)#使用训练集词袋向量
    #该类会统计每个词语的TF-IDF权值
    transformer = TfidfTransformer()
    #文本转为词频矩阵，单独保存字典文件
    testspace.tdm = vectorizer.fit_transform(bunch.contents)
    testspace.vocabulary = vectorizer.vocabulary
    #持久化TF-IDF向量词袋
    space_path = "D:/materials/dataset/test_word_bag/testspace.dat"
    writebunchobj(space_path,testspace)

#######################################执行多项式贝叶斯算法进行测试文本分类，并返回分类精度#####################################
from sklearn.naive_bayes import MultinomialNB #导入多项式贝叶斯算法包
from sklearn import metrics
def get_category():
    trainpath = "D:/materials/dataset/train_word_bag/tfidfspace.dat" #导入训练集向量空间
    train_set = readbunchobj(trainpath)

    testpath = "D:/materials/dataset/test_word_bag/testspace.dat" #导入测试集向量空间
    test_set = readbunchobj(testpath)

    clf = MultinomialNB(alpha=0.001).fit(train_set.tdm,train_set.label) #应用贝叶斯算法，slpha越小，迭代次数越高，精度越高
    predicted = clf.predict(test_set.tdm) #预测分类结果
    total = len(predicted)
    rate = 0
    for flabel,file_name,expect_cate in zip(test_set.label,test_set.filenames,predicted):
        if flabel != expect_cate:
            rate +=1
            print file_name,": 实际类别：",flabel,"-->预测类别：",expect_cate
    print "error rate:",float(rate )*100/float(total),"%"
    #分类精度
    print "精度：{0:.3f}".format(metrics.predision_score(test_set.label,predicted))
    print "召回: {0:.3f}".format(metrics.recall_score(test_set.label,predicted))
    print "f1-score:{0:.3f}".format(metrics.f1_score(test_set.label,predicted))

# if __name__ == '__main__':
    # participle_test()
    # persistence_to_Bunch_test()
    # get_tfidf_test()
    # get_category()