from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from tokenizers import Tokenizer
from tokenizers import BertWordPieceTokenizer

def train_BPE_tokenizer(vocab_size):
    special_tokens = ["<pad>","<unk>","<s>","</s>","<mask>"]
    tokenizer = Tokenizer(BPE(unk_token="<unk>", sep_token="</s>", mask_token="<mask>",cls_token="<s>", pad_token="<pad>"))
    file_dir = '../raw_reddit_dedup' 
    files_reddit = [os.path.join(file_dir,filename) for filename in os.listdir(file_dir)]
    print(f"reddit has file {len(files_reddit)}")
    file_dir = '../raw_twitter_dedup'
    files_twitter = [os.path.join(file_dir,filename) for filename in os.listdir(file_dir)]
    print(f"twitter has file {len(files_twitter)}")
    file_dir = '../raw_ccnews_dedup'
    files_ccnews = [os.path.join(file_dir,filename) for filename in os.listdir(file_dir)]
    print(f"ccnews has file {len(files_ccnews)}")
    trainer = BpeTrainer(special_tokens=special_tokens, min_frequency = 0, vocab_size = vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    files = (files_reddit + files_twitter + files_ccnews)
    tokenizer.train(files, trainer)
    save_name = 'tokenizer_merge.json'
    tokenizer.save(save_name)

def train_WP_tokenizer():
    pass

def cal_combine_vocab():
    merge_vocab = set()
    tokenizer_list = ['tokenizer_.json', 'tokenizer_twitter_reddit.json', 'tokenizer_twitter_reddit_ccnews.json']

    for tokenizer_name in tokenizer_list:
        print(f"processing tokenizer {tokenizer_name}...")
        cur_tokenizer = Tokenizer.from_file(tokenizer_name)
        for word,idx in cur_tokenizer.get_vocab().items():
            merge_vocab.add(word)
    print(f"Finish processing.\nTotal vocab size is {len(merge_vocab)}...")
    return len(merge_vocab)


vocab_size = cal_combine_vocab()
train_BPE_tokenizer(vocab_size)

