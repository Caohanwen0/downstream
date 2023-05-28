import torch, os, json, csv
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import bmtrain as bmt
from cmath import inf
from model_center.model import Roberta,RobertaConfig
from model_center.layer import Linear
from model_center.dataset.bertdataset import DATASET
from model_center.dataset import DistributedDataLoader
from model_center.dataset import MMapIndexedDataset, DistributedMMapIndexedDataset, DistributedDataLoader
from model_center.tokenizer import RobertaTokenizer
from model_center.utils import print_inspect
from transformers import AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
import pandas as pd
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
    bmt.init_distributed(seed = args.seed, zero_level=2,)
    # init save folder
    if args.save != None:
        os.makedirs(args.save , exist_ok=True)
    bmt.print_rank(args)
    return args

args = initialize()

if bmt.rank() == 0:
    if args.save_tensorboard == True:
        writer = SummaryWriter(os.path.join(args.save, 'tensorborads'))

bmt.print_rank(f"Local rank:{bmt.rank()} | World size:{bmt.world_size()}")

train_texts, train_labels = parse_data(args, "train")
val_texts, val_labels = parse_data(args, "validation")
test_texts, test_labels = parse_data(args, "test")

label_num = len(set(test_labels))

bmt.print_rank(f"Train size: {len(train_labels)} | Val size: {len(val_labels)} | Test size: {len(test_labels)}")
bmt.print_rank(f"Fine-tuning: {label_num} class classification...")

class MyRobertaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if args.checkpoint is not None:
            bmt.print_rank(f"Initializing roberta from pretrained {args.checkpoint}...")
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
bmt.synchronize()

def get_tokenizer():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.unk_token = '<unk>'
    tokenizer.mask_token = '<mask>'
    tokenizer.cls_token = '<s>'
    tokenizer.sep_token = '<\s>'
    #tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)
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

optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=1e-2, betas = (0.9, 0.98))
# optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters())
lr_scheduler = bmt.lr_scheduler.Linear(
    optimizer, 
    start_lr = args.lr,
    warmup_iter = args.warmup_ratio * total_step,  # default to 0
    end_iter = total_step,
    num_iter=0)

optim_manager = bmt.optim.OptimManager(loss_scale = 2**20)
# add_optimizer can be called multiple times to add other optimizers.
optim_manager.add_optimizer(optimizer, lr_scheduler)
bmt.synchronize()

loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

def remove_last_save_model():
    l = os.listdir(args.save)
    for filename in l:
        if filename[:11] == "checkpoint-":
            if os.path.isfile(os.path.join(args.save, filename)):
                if bmt.rank() == 0:
                    os.remove(os.path.join(args.save, filename))
                    bmt.print_rank(f"Removing previous checkpoint {filename}...")

def get_last_epoch():
    return 0
    l = os.listdir(args.save)
    last_epoch = -1
    load = False
    for filename in l:
        if filename[:11] == "checkpoint-":
            load = True
            cur_last_epoch = int(filename.split('.')[0][11:].split('-')[0])
            if cur_last_epoch > last_epoch:
                last_epoch = cur_last_epoch
    bmt.print_rank(f"Resume training at epoch {last_epoch + 2}...")
    if load:
        bmt.load(model, os.path.join(args.save, f'checkpoint-{last_epoch}.pt'))
    bmt.synchronize()
    return last_epoch
    
def fine_tune():
    best_valid_acc = 0
    global_step = 0
    early_stopping = 0
    last_epoch = -1
    remove_last_save_model()
    # last_epoch = get_last_epoch()
    for epoch in range(last_epoch + 1, args.epochs):
        bmt.print_rank("Epoch {} begin...".format(epoch + 1))
        model.train()
        for step, data in enumerate(train_dataloader):
            global_step += 1
            if global_step % args.gradient_accumulate == 1:
                optim_manager.zero_grad() # calling zero_grad for each optimizer
            input_ids, attention_mask, labels = data
            # load to cuda
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            global_loss = bmt.sum_loss(loss).item()
            if bmt.rank() == 0:
                if args.save_tensorboard == True:
                    writer.add_scalar(f"Loss/train", global_loss, global_step)
            # loss = optimizer.loss_scale(loss)
            # loss.backward()
            loss = loss / args.gradient_accumulate
            optim_manager.backward(loss)
            if global_step % args.gradient_accumulate == 0:
                grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm = 1.0 , norm_type = 2)
                optim_manager.step()
            # bmt.optim_step(optimizer, lr_scheduler)
            if step % args.log_iters == 0 and (not step == 0):
                bmt.print_rank(
                    "Loss: {:.4f} | Scale: {:10.4f} | Grad_norm: {:.4f} | Lr: {:.4e}".format(
                        global_loss,
                        optim_manager.loss_scale,
                        grad_norm,
                        lr_scheduler.current_lr,
                    )
                )
        # evaluate model on dev dataset
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
            bmt.print_rank(f"Validation accuracy: {acc*100:.2f}\n")
            if acc > best_valid_acc:
                best_valid_acc = acc
                bmt.print_rank("Saving the new best model...\n") # save checkpoint
                early_stopping = 0
                remove_last_save_model()
                bmt.save(model, os.path.join(args.save,  f'checkpoint-{epoch}.pt'))
            if early_stopping == 5:
                bmt.print_rank("Accuracy have not rising for 5 epochs. Early stopping..")
                break # break for iter

def load_saved_model():
    l = os.listdir(args.save)
    for filename in l:
        if filename[:11] == "checkpoint-":
            if os.path.isfile(os.path.join(args.save, filename)):
                bmt.load(model, os.path.join(args.save, filename))
                return True
    return False 

def check_performance():
    load_succeed = load_saved_model()
    if not load_succeed:
        exit(1)
    bmt.synchronize()
    bmt.print_rank(f"Checking performance of dataset {args.dataset_name} with learning rate {args.lr}...\n")
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
    # epoch_num = fine_tune()
    check_performance()