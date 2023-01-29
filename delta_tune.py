from cmath import inf
import bmtrain as bmt
bmt.init_distributed(seed=0)
import torch
from model_center.model import Bert, BertConfig,Roberta,RobertaConfig
from model_center.layer import Linear
from model_center.dataset.bertdataset import DATASET
from model_center.dataset import DistributedDataLoader
from model_center.dataset import MMapIndexedDataset, DistributedMMapIndexedDataset, DistributedDataLoader
from model_center.tokenizer import BertTokenizer, RobertaTokenizer
from model_center.utils import print_inspect
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import pandas as pd
import json,os,csv
from tqdm import tqdm
from arguments import get_args
from torch.utils.tensorboard import SummaryWriter
from opendelta import AdapterModel


# bmt.print_rank("torch version", torch.__version__)
# bmt.print_rank(torch.cuda.get_arch_list())


def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save , exist_ok=True)
    return args

args = initialize()
print(args)
if bmt.rank() == 0:
    writer = SummaryWriter(os.path.join(args.save, 'tensorborads'))

bmt.print_rank(f"Local rank:{bmt.rank()} | World size:{bmt.world_size()}")

def get_dataset(csvfile):
    dataset_path = os.path.join(args.dataset_path, args.dataset_name)
    lines = None
    texts, labels = [], []
    with open(os.path.join(dataset_path, csvfile), newline = '')as file:
        reader = csv.reader(file)
    del reader[0]
    for row in reader:
        texts.append(row[0])
        labels.append(int(row[1]))
    print(f"Sample:\nText: {texts[0]}\n:Label: {labels[0]}")

    

train_texts, train_labels = get_dataset("test.csv")
val_texts, val_labels = get_dataset("validation.csv")
test_texts, test_labels = get_dataset("train.csv")

label_num = len(set(test_labels))

def get_model():
    config = RobertaConfig.from_json_file(args.model_config)
    model = RobertaModel(config)
    delta_model = AdapterModel(backbone_model = model, modified_modules = [
        'self_att', 'ffn',
    ], backend = 'bmt')
    delta_model.freeze_module(exclude = ['deltas'], set_state_dict = True)
    delta_model.log()
    return delta_model

model = get_model()

bmt.print_rank(f"Train size: {len(train_labels)} | Val size: {len(val_labels)} | Test size: {len(test_labels)}")
bmt.print_rank(f"{label_num} class classification...")

def get_tokenzer():
    tokenizer_obj = Tokenizer.from_file(args.tokenizer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '</s>'
    tokenizer.sep_token = '<s>'
    tokenizer.unk_token = '<unk>'
    tokenizer.mask_token = '<mask>'

tokenizer = get_tokenizer()

tokens_train = tokenizer.batch_encode_plus(
    train_texts,
    max_length = args.padding_len,
    padding='max_length',
    truncation=True,
    add_special_tokens = True,
)

tokens_val = tokenizer.batch_encode_plus(
    val_texts,
    max_length = args.padding_len,
    padding='max_length',
    truncation=True,
    add_special_tokens = True,
)

tokens_test = tokenizer.batch_encode_plus(
    test_texts,
    max_length = args.padding_len,
    padding='max_length',
    truncation=True,
    add_special_tokens = True,
)

train_data = TensorDataset(torch.tensor(tokens_train['input_ids']), \
    torch.tensor(tokens_train['attention_mask']), \
    torch.tensor(train_labels))
val_data = TensorDataset(torch.tensor(tokens_val['input_ids']), \
    torch.tensor(tokens_val['attention_mask']), \
    torch.tensor(val_labels))
test_data = TensorDataset(torch.tensor(tokens_test['input_ids']), \
    torch.tensor(tokens_test['attention_mask']), \
    torch.tensor(test_labels))

train_dataloader = DistributedDataLoader(train_data, batch_size = args.batch_size, shuffle = True)
val_dataloader = DistributedDataLoader(val_data, batch_size = args.batch_size, shuffle = False)
test_dataloader = DistributedDataLoader(test_data, batch_size = args.batch_size, shuffle = False)

# optimizer and lr-scheduler
total_step = (len(train_dataloader)) * args.epochs
optimizer = bmt.optim.AdamOptimizer(model.parameters(),lr = args.lr, betas=(0.9,0.98))
# optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters())
lr_scheduler = bmt.lr_scheduler.Linear(
    optimizer, 
    start_lr = 0,
    warmup_iter = args.warmup_ratio * total_step,  # default to 0
    end_iter = total_step)

loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    
def fine_tune():
    best_valid_acc = 0
    valid_acc_list = []
    early_stopping = 0
    global_step = 0
    for epoch in range(args.epochs):
        bmt.print_rank("Epoch {} begin...".format(epoch + 1))
        model.train()
        for step, data in enumerate(train_dataloader):
            global_step += 1
            input_ids, attention_mask, labels = data
            # load to cuda
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            global_loss = bmt.sum_loss(loss).item()
            if bmt.rank() == 0:
                writer.add_scalar(f"Loss/train", global_loss, global_step)
            loss = optimizer.loss_scale(loss)
            loss.backward()
            grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups,max_norm= float('inf'), scale = optimizer.scale, norm_type = 2)
            bmt.optim_step(optimizer, lr_scheduler)
            #bmt.optim_step(optimizer) # fixed learning rate
            if step % args.inspect_iters == 0:
                bmt.print_rank(
                    "loss: {:.4f} | scale: {:10.4f} | grad_norm: {:.4f} |".format(
                        global_loss,
                        int(optimizer.scale),
                        grad_norm,
                    )
                )
        model.eval()
        with torch.no_grad():
            pd = [] # prediction
            gt = [] # ground_truth
            for data in val_dataloader:
                input_ids, attention_mask, labels = data
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
                logits = model(input_ids, attention_mask)
                loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
                logits = logits.argmax(dim=-1)
                pd.extend(logits.cpu().tolist())
                gt.extend(labels.cpu().tolist())

            # gather results from all distributed processes
            pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
            gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

            # calculate metric
            acc = accuracy_score(gt, pd)
            if bmt.rank()==0:
                writer.add_scalar(f"Acc/dev", acc, epoch)
            early_stopping += 1
            bmt.print_rank(f"validation accuracy: {acc*100:.2f}\n")
            if acc > best_valid_acc:
                best_valid_acc = acc
                bmt.print_rank("saving the new best model...\n") # save checkpoint
                bmt.save(model, os.path.join(args.save,  'model.pt'))
                early_stopping = 0 
            valid_acc_list.append(acc)
            # if early_stopping > 5:
            #     bmt.print_rank("Accuracy have not rising for 5 epochs.Early stopping..")
            #     break # break for iter

def check_performance():
    bmt.load(model, os.path.join(args.save, 'model.pt'))
    bmt.print_rank("Checking performance...\n")
    with torch.no_grad():
        pd = [] # prediction
        gt = [] # ground_truth
        for data in test_dataloader:
            input_ids, attention_mask, labels = data
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            logits = logits.argmax(dim=-1)

            pd.extend(logits.cpu().tolist())
            gt.extend(labels.cpu().tolist())

        # gather results from all distributed processes
        pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
        gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

        bmt.print_rank(classification_report(y_true = gt, y_pred = pd, digits = 5))

fine_tune()
check_performance()