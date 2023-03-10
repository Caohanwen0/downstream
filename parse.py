import os, csv, logging

from utils import *

def parse_data(args, type):
    category = args.dataset_name.split("/")[0]
    datasetname = args.dataset_name.split("/")[1]
    assert (type in ["train", "validation", "test"])
    lines = None
    if os.path.isfile(os.path.join(args.dataset_path, args.dataset_name, 'train.txt')):
        with open(os.path.join(args.dataset_path, args.dataset_name, f"{type}.txt")) as fin:
            lines = fin.readlines()
    texts, labels = [], []

    if category == "sentiment":
        if datasetname == "semeval2017task4a":
            label_to_int= {
                "positive" : 0,
                "neutral" : 1,
                "negative" : 2}
            for step, line in enumerate(lines):
                texts.append(preprocess_clean(line.split("	")[0]))
                label = label_to_int[line.split("	")[1].strip("\n")]
                labels.append(label)
        elif datasetname == "twitter_US_airline_dataset":
            label_to_int= {
                "positive" : 0,
                "neutral" : 1,
                "negative" : 2
            }
            for line in lines:
                texts.append(preprocess_clean(line.split("	")[0]))
                label = label_to_int[line.split("	")[1].strip("\n")]
                labels.append(label)
        elif datasetname == "NewsMTSC":
            # 2.0: 0, 4.0: 1, 6.0: 2
            with open(os.path.join(args.dataset_path, args.dataset_name, f"{type}.csv")) as fin:
                csvreader = csv.reader(fin)
                for step, row in enumerate(csvreader):
                    if step == 0:
                        continue
                    texts.append(preprocess_clean(row[1]))
                    if int(row[-1]) == 1:
                        labels.append(2)
                    elif int(row[-2]) == 1:
                        labels.append(1)
                    elif int(row[-3]) == 1:
                        labels.append(0)
                    else:
                        logging.warning("Parsing dataset NewsMTSC error!")
                        exit(1)
    elif category == "emotion":
        if datasetname == "emoint":
            label_to_int = {
                "anger":0,
                "fear":1,
                "sadness":2,
                "joy":3
            }
            for line in lines:
                texts.append(preprocess_clean(line.split("	")[0]))
                label = label_to_int[line.split("	")[1].strip("\n")]
                labels.append(label)
    elif category == "stance":
        if datasetname == "semeval2016task6":
            # favor : 0, none : 1 , against : 2
            with open(os.path.join(args.dataset_path, args.dataset_name, f"{type}.csv")) as fin:
                csvreader = csv.reader(fin)
                for step, row in enumerate(csvreader):
                    if step == 0:
                        continue
                    texts.append(preprocess_clean(row[2]))
                    if int(row[-1]) == 1:
                        labels.append(1)
                    elif int(row[-2]) == 1:
                        labels.append(2)
                    elif int(row[-3]) == 1:
                        labels.append(0)
                    else:
                        logging.warning("Parsing stance dataset semeval 2016 task6 error!")
                        exit(1) 
    elif category == "offensive":
        if datasetname == "reddit_incivility":
            # incivility : 1 , not-incivility: 0
            with open(os.path.join(args.dataset_path, args.dataset_name, f"{type}.csv")) as fin:
                csvreader = csv.reader(fin)
                for step, row in enumerate(csvreader):
                    if step == 0:
                        continue
                    texts.append(preprocess_clean(row[1]))
                    try:
                        labels.append(int(row[-1]))
                    except:
                        logging.warning(f"An error had occuredat at line  : {row}")
        elif datasetname == "OLID":
            label_to_int = {
                "NOT": 0,
                "OFF" :1
            }
            for line in lines:
                line = line.split("	")
                texts.append(preprocess_clean(line[0])) 
                labels.append(label_to_int[line[1].strip("\n")])
        elif datasetname == "COLD":
            with open(os.path.join(args.dataset_path, args.dataset_name, f"{type}.csv")) as fin:
                csvreader = csv.reader(fin)
                for step, row in enumerate(csvreader):
                    if step == 0:
                        continue
                    texts.append(preprocess_clean(row[1]))
                    try:
                        false_label = int(row[2]) # non-offensive
                        true_label = int(row[3])
                        assert false_label + true_label == 1
                        labels.append(true_label)
                    except:
                        logging.warning(f"An error had occuredat at line  : {row}")
        
    elif category == "rumor":
        if datasetname == "Weibo20":
            with open(os.path.join(args.dataset_path, args.dataset_name, f"{type}.csv")) as fin:
                csvreader = csv.reader(fin)
                for step, row in enumerate(csvreader):
                    if step == 0:
                        continue
                    texts.append(preprocess_clean(row[1]))
                    try:
                        false_label = int(row[2])
                        true_label = int(row[3])
                        assert false_label + true_label == 1
                        labels.append(true_label)
                    except:
                        logging.warning(f"An error had occuredat at line  : {row}")
        
    assert len(texts) == len(labels)
    return texts, labels

class MyArgs:
    def __init__(self):
        self.dataset_path = "datasets"
        self.dataset_name = "offensive/OLID"
    

if __name__ == "__main__":
    args = MyArgs()
    t, l = parse_data(args, "train")
    print(f"Dataset size : {len(t)}")