import torch

import bmtrain as bmt
from cmath import inf
from model_center.model import Roberta,RobertaConfig
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
from parse import parse_data


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
    if args.save_tensorboard == True:
        writer = SummaryWriter(os.path.join(args.save, 'tensorborads'))

bmt.print_rank(f"Local rank:{bmt.rank()} | World size:{bmt.world_size()}")


train_texts, train_labels = parse_data(args, "train")
val_texts, val_labels = parse_data(args, "validation")
test_texts, test_labels = parse_data(args, "test")

label_num = len(set(test_labels))

class MyRobertaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if args.checkpoint is not None:
            bmt.print_rank(f"Initializing bert from pretrained {args.checkpoint}...")
            self.roberta = Roberta.from_pretrained(args.checkpoint) # load roberta from pretrained
        else: 
            assert args.load is not None
            bmt.print_rank(f"Initializing roberta from our model path {args.load}...")
            self.roberta = Roberta(config)
            bmt.load(self.roberta, args.load, strict = False)
        # print_inspect(self.roberta, "*")
        self.dense = Linear(config.dim_model, label_num)
        bmt.init_parameters(self.dense) # init dense layer

    def reload(self, config):
        super().__init__()
        if args.checkpoint is not None:
            bmt.print_rank(f"Initializing bert from pretrained {args.checkpoint}...")
            self.roberta = Roberta.from_pretrained(args.checkpoint) # load roberta from pretrained
        else: 
            assert args.load is not None
            bmt.print_rank(f"Initializing bert from our model path {args.load}...")
            self.roberta = Roberta(config)
            bmt.load(self.roberta, args.load)
        print_inspect(self.roberta, "*")
        self.dense = Linear(config.dim_model, label_num)
        bmt.init_parameters(self.dense) # init dense layer 

    def forward(self, input_ids, attention_mask):
        pooler_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.dense(pooler_output)
        return logits

def get_model():
    config = RobertaConfig.from_json_file(args.model_config)
    roberta_model = MyRobertaModel(config)
    return roberta_model

model = get_model()

bmt.print_rank(f"Train size: {len(train_labels)} | Val size: {len(val_labels)} | Test size: {len(test_labels)}")
bmt.print_rank(f"Fine-tuning: {label_num} class classification...")

def get_tokenizer():
    tokenizer_obj = Tokenizer.from_file(os.path.join(args.base_path, "downstream", args.tokenizer))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = tokenizer_obj)
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '</s>'
    tokenizer.sep_token = '<s>'
    tokenizer.unk_token = '<unk>'
    tokenizer.mask_token = '<mask>'
    return tokenizer

tokenizer = get_tokenizer()

tokens_train = tokenizer.batch_encode_plus(
    train_texts,
    max_length = args.max_length,
    padding='max_length',
    truncation=True,
    add_special_tokens = True,
)

tokens_val = tokenizer.batch_encode_plus(
    val_texts,
    max_length = args.max_length,
    padding='max_length',
    truncation=True,
    add_special_tokens = True,
)

tokens_test = tokenizer.batch_encode_plus(
    test_texts,
    max_length = args.max_length,
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
                if args.save_tensorboard == True:
                    writer.add_scalar(f"Loss/train", global_loss, global_step)
            loss = optimizer.loss_scale(loss)
            loss.backward()
            grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups,max_norm= float('inf'), scale = optimizer.scale, norm_type = 2)
            bmt.optim_step(optimizer, lr_scheduler)
            #bmt.optim_step(optimizer) # fixed learning rate
            if step % args.log_iters == 0:
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
                if args.save_tensorboard == True:
                    writer.add_scalar(f"Acc/dev", acc, epoch)
            early_stopping += 1
            bmt.print_rank(f"validation accuracy: {acc*100:.2f}\n")
            if acc > best_valid_acc:
                best_valid_acc = acc
                bmt.print_rank("saving the new best model...\n") # save checkpoint
                bmt.save(model, os.path.join(args.save,  'model.pt'))
                early_stopping = 0 
            valid_acc_list.append(acc)
            if early_stopping > 5:
                bmt.print_rank("Accuracy have not rising for 5 epochs.Early stopping..")
                break # break for iter

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


if __name__ == "__main__":
    fine_tune()
    check_performance()