import torch
import torch.nn as nn
import pandas as pd
import codecs
import csv,os
from probability import session_probability
from SetSeed import setseed
import torch.optim as optim
from transformers import BertModel, AutoModel
from transformers import BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import Dataset
import numpy as np

class Dataprocessor(Dataset):

    def __init__(self, filename, maxlen):

        self.filename=filename
        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')
        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.filename=='test.tsv':
            sentence = self.df.loc[index, 'sentence']
        else:
            # Selecting the sentence and label at the specified index in the data frame
            sentence = self.df.loc[index, 'sentence']
            label = self.df.loc[index, 'label']
            # sentence = self.df.iloc[1]
            # label = self.df.iloc[2]

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence)  # Tokenize the sentence
        # Insering the CLS and SEP token in the beginning and end of the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            # Padding sentences
            tokens = tokens + \
                     ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            # Prunning the list to be of specified max length
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']

        # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # Converting the list to a pytorch tensor
        tokens_ids_tensor = torch.tensor(tokens_ids)

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()
        if self.filename=='test.tsv':
            return tokens_ids_tensor, attn_mask
        else:
            return tokens_ids_tensor, attn_mask,label



class SentimentClassifier(nn.Module):

    def __init__(self, freeze_bert=True, num_labels=2):
        super(SentimentClassifier, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = AutoModel.from_pretrained(model_path)
        self.num_labels = num_labels
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, self.num_labels)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask=attn_masks)

        # Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits


def evaluate(net, criterion, dataloader, num_labels):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, labels) in enumerate(dataloader):
            if isCuda:
                seq, attn_masks, labels = seq.cuda('cuda:0'), attn_masks.cuda(
                    'cuda:0'), labels.cuda('cuda:0')
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.view(-1, num_labels), labels.view(-1)).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1
    return mean_acc / count, mean_loss / count


def train(net, criterion, opti, train_loader, val_loader, num_labels,epoches):
    for ep in range(epoches):
        best_acc = 0
        # single = len(train_loader)
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            if isCuda:
                seq, attn_masks, labels = seq.cuda('cuda:0'), attn_masks.cuda(
                    'cuda:0'), labels.cuda('cuda:0')

            # Obtaining the logits from the model
            logits = net(seq, attn_masks)
            # Computing loss
            loss = criterion(logits.view(-1, num_labels), labels.view(-1).long())

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            opti.step()
            if (it + 1) % 10 == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(
                    it + 1, ep + 1, loss.item(), acc))

        val_acc, val_loss = evaluate(net, criterion, val_loader, num_labels)
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(
            ep, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(
                best_acc, val_acc))
            best_acc = val_acc
            torch.save(net.state_dict(),
                       'Models/sstcls_{}_freeze_{}.bin'.format(ep, False))

def predict(net,opti,test_loader):
    preds=None
    for it, (seq, attn_masks,_) in enumerate(test_loader):
        # Clear gradients
        opti.zero_grad()
        # Converting these to cuda tensors
        if isCuda:
            seq, attn_masks = seq.cuda('cuda:0'), attn_masks.cuda(
                'cuda:0')

        # Obtaining the logits from the model
        logits = net(seq, attn_masks)
        print(logits.detach().cpu().numpy())
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds,logits.detach().cpu().numpy(),axis=0)
            print(preds)
        print("logits:",logits.detach().cpu().numpy())
    write_list = session_probability(data_path,preds,1)
    # write_list = session_probability(写死,preds,1)
    output_pred_file = os.path.join(pred_output_dir, "test_prediction.txt")
    with open(output_pred_file, "w") as writer:
        writer.write('SessionId,Probability\n')
        for id,element in enumerate(write_list):
            writer.write(str(id) + "," + str(element)+'\n')

def create_labels(li_exam):
    examples = []
    labels = []
    for index, ele in enumerate(li_exam):
        if index == 0:
            continue
        label = int(ele[1])
        labels.append(label)
    return set(labels)


def read_csv(input_file):
    li_exam = []
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            print("171line[0]",line[0])
            line = list(line[0].split('\t'))
            li_exam.append(line)
    return li_exam


def get_accuracy_from_logits(logits, labels):
    probs = logits.unsqueeze(-1)
    soft_probs = torch.argmax(probs, 1).view(1, -1).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


if __name__ == "__main__":
    # model的路径
    model_path = './pretrained_model/'
    # 数据的路径
    data_path = './data/'
    #预测保存目录
    pred_output_dir='./pred_output_dir/'
    # 训练轮数
    epoches = 1
    # batchsize
    train_batch_size = 32
    test_batch_size = 128

    # 判断cuda是否可用
    isCuda = torch.cuda.is_available()
    setseed(777)

    # Creating instances of training and validation set
    labels = [0, 1]
    train_set = Dataprocessor(filename=data_path + 'train.tsv', maxlen=30)
    val_set = Dataprocessor(filename=data_path + 'dev.tsv', maxlen=30)
    test_set=Dataprocessor(filename=data_path+'test.tsv',maxlen=30)

    # Creating intsances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=train_batch_size, num_workers=5, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, num_workers=5)
   # pred_sampler = SequentialSampler(test_set)
    test_loader=DataLoader(test_set,batch_size=train_batch_size,num_workers=5)
    net = SentimentClassifier(freeze_bert=False, num_labels=len(labels))

    if isCuda:
        net.cuda('cuda:0')

    criterion = nn.CrossEntropyLoss()
    opti = optim.Adam(net.parameters(), lr=2e-5)

    train(net, criterion, opti, train_loader, val_loader, num_labels=len(labels),epoches=epoches)
    predict(net,opti,test_loader)
