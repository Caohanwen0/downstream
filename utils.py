from emoji import demojize
from nltk.tokenize import TweetTokenizer
import re
import html
import opencc
t2s_convert=opencc.OpenCC('t2s.json')
tokenizer = TweetTokenizer()

with open('src/change.txt') as f:
    mapping = {}
    for line in f:
        chars = line.strip('\n').split('\t')
        mapping[chars[0]] = chars[1]
    # 替换回车键至换行键
    mapping["\u000D"] = "\u000A"
    mapping["\u2028"] = "\u000A"
    mapping["\u2029"] = "\u000A"
    # 替换\t至空格
    mapping["\u0009"] = "\u0020"

with open("src/yiti_ch_change.txt") as f:
    yiti_change = {}
    for line in f:
        chars = line.strip('\n').split('\t')
        yiti_change[chars[0]] = chars[1]

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

def isChineseChar(char):
    '''
    判断一个字符是否为汉字
    '''
    chinese_char = [['\u2E80', '\u2EFF'],
                    ['\u2F00', '\u2FDF'],
                    ['\u3400', '\u4DBF'],
                    ['\u4E00', '\u9FFF'],
                    ['\uF900', '\uFAFF'],
                    ['\U00020000', '\U0002A6DF'],
                    ['\U0002A700', '\U0002B73F'],
                    ['\U0002B740', '\U0002B81F'],
                    ['\U0002B820', '\U0002CEAF'],
                    ['\U0002CEB0', '\U0002EBEF'],
                    ['\U0002F800', '\U0002FA1F'],
                    ['\U00030000', '\U0003134F']]

    for line in chinese_char:
        if line[0] <= char <= line[1]:
            return True
    return False

# 正则
def repl1(matchobj):
    return matchobj.group(0)[0]
def repl2(matchobj):
    return matchobj.group(0)[-1]

def preprocess_clean(sent):
    '''
    进一步的预处理
    英文步骤包括：字符替换、空白字符删除、标点符号清洗；
    社交媒体特殊处理：@username -> '@USER' http链接 -> 'HTTPURL'
    '''
    # 建立替换表

    '''
    替换特殊字符以及删除不可见字符
    ''' 

    # 繁简转换
    try:
        sent=t2s_convert.convert(sent)
    except:
        print("fan2jan error! ", sent)
    # 修正不会被替换的        
    fix_dict={'甚么':'什么','妳':'你'}
    for x,y in fix_dict.items():
        sent=sent.replace(x,y)

    # 字符转换
    sent = ''.join(map(lambda x:mapping.get(x, x), sent))
    # 异体字转换
    sent = ''.join(map(lambda x:yiti_change.get(x, x), sent))
    # 去除不可见字符
    sent = ''.join(x for x in sent if x.isprintable())

    # 标点重复
    sent=re.sub(r'([（《【‘“\(\<\[\{）》】’”\)\>\]\} ,;:·；：、，。])\1+',repl1,sent)
    # 括号紧跟标点
    sent=re.sub(r'[（《【‘“\(\<\[\{][ ,.;:；：、，。！？·]',repl1,sent)
    sent=re.sub(r'[ ,.;:；：、，。！？·][）》】\)\>\]\}]',repl2,sent)
    # 括号内为空
    sent=re.sub(r'([（《【‘“\(\<\[\{\'\"][\'\"）》】’”\)\>\]\}])','',sent)
    # 三个。和.以上的转为...
    sent = re.sub(r'[。.]{3,}', '...', sent)
    # HTML网址清洗和username清洗
    sent = re.sub("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "HTTPURL", sent)      # url
    sent = re.sub("@\S+", "@USER", sent)      # username
    # html_entities 转译
    sent = html.unescape(sent)

    # 删除句内汉字与其他符号之间的空格
    char_list=list(sent) 
    length = len(char_list)
    last = True
    for id, x in enumerate(char_list):
        if x==' ':
            if last or ((id+1<length)and(isChineseChar(char_list[id+1]) or char_list[id+1]==' ')):
                char_list[id]=''
            else:
                last = False
        else:
            if x in '。，：；！？【】《》“”':
                last = True
            else:
                last = isChineseChar(x)

    result_sent = "".join(char_list) 
    return result_sent