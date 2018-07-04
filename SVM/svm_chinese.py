#encoding=utf-8

import sys
import pickle

reload(sys)
sys.setdefaultencoding('utf-8')

def readbunchobj(path):
    file_obj = open(path,"rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch
def writebunchobj(path,bunchobj):
    file_obj = open(path,"wb")
    pickle.dump(bunchobj,file_obj)
    file_obj.close()

#######################################执行SVM算法进行测试文本分类#####################################
from sklearn.svm import LinearSVC #导入线性VM算法
from sklearn import metrics
def get_category():
    trainpath = "D:/materials/dataset/naivebayes_chinese_dataset/train_word_bag/tfidfspace.dat" #导入训练集向量空间
    train_set = readbunchobj(trainpath)
    testpath = "D:/materials/dataset/naivebayes_chinese_dataset/test_word_bag/testspace.dat" #导入测试集向量空间
    test_set = readbunchobj(testpath)

    #1.输入词袋向量和分类标签
    clf = LinearSVC(penalty='l2',dual=False,tol=1e-4).fit(train_set.tdm,train_set.label)
    #2.预测分类结果
    predicted = clf.predict(test_set.tdm)
    total = len(predicted)
    rate = 0
    for flabel,file_name,expect_cate in zip(test_set.label,test_set.filenames,predicted):
        if flabel != expect_cate:
            rate +=1
            print file_name,": 实际类别：",flabel,"-->预测类别：",expect_cate
    print "error rate:",float(rate )*100/float(total),"%"

    # 分类精度
    print "精度：{0:.3f}".format(metrics.predision_score(test_set.label, predicted))
    print "召回: {0:.3f}".format(metrics.recall_score(test_set.label, predicted))
    print "f1-score:{0:.3f}".format(metrics.f1_score(test_set.label, predicted))

if __name__ == '__main__':
    get_category()