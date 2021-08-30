# -8*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import argparse, random, time, os
import numpy as np

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.docs = torch.randn((1024, 32, 16))
    def __len__(self):
        return len(self.docs)
    def __getitem__(self, index) :
        return self.docs[index]

class MyModel(nn.Module):
    def __init__(self, max_seq_len=32, emb_dim=16):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.position_layer = nn.Embedding(max_seq_len, emb_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=2, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(emb_dim, 4)
    def forward(self, imgs, mask):
        postions = self.position_layer(torch.arange(self.max_seq_len).repeat((imgs.shape[0], 1)).to(imgs).long())
        imgs = imgs + postions
        feature = self.encoder(imgs, src_key_padding_mask=~mask)
        pooling1 = torch.sum((feature * mask.unsqueeze(-1)), axis=1) / mask.sum(axis=1)
        pooling2 = torch.max((feature * mask.unsqueeze(-1)), axis=1)[0]
        pooling = torch.cat([pooling1, pooling2], dim=1)
        output = self.fc(pooling)
        return output

class Trainer():
    def __init__(self, model, dataloader, datasampler, device, rank, args):
        self.model = model
        self.dataloader = dataloader
        self.datasampler = datasampler
        self.device = device
        self.rank = rank
        self.args = args
    def _data_to_gpu(self, data, device):
        for k in data:
            data[k] = torch.tensor(data[k]).to(device)
        return data
    def predict(self, dataloader=None, is_valid=False):
        y_true, y_pred = [], []
        self.model.eval()
        if dataloader is None:
            dataloader = self.dataloader
        with torch.no_grad():
            for batch in dataloader:
                input = [self._data_to_gpu(data, self.device) for data in batch]
                if is_valid:
                    feature, label = input[:-1], input[-1]
                else:
                    feature, label = input[:-1], None
                output = self.model(feature)
                predicted_label = torch.argmax(output, dim=1).detach().cpu().numpy().tolist()
                y_pred += predicted_label
                y_true += [0] * len(predicted_label) if not is_valid else label.detach().cpu().numpy().tolist()
        self.model.eval()
        return y_true, y_pred

    def fit(self, epoch, optimizer, criterion, saved_model, scheduler=None, validloader=None):
        for epoch in range(1, epoch+1):
            time1 = time.time()
            self.model.train(True)
            self.datasampler.set_epoch(epoch)
            total_loss = []

            for batch in self.dataloader: 
                optimizer.zero_grad()

                input = [self._data_to_gpu(data, self.device) for data in batch]
                feature, label = input[:-1], input[-1]

                output = self.model(feature)
                loss = criterion(output, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)

                optimizer.step()

                if self.rank == 0:
                    total_loss.append(loss.item())
            
            if self.rank == 0:
                epoch_avg_loss = np.mean(total_loss)
                print("Epoch {:02d}, Time {:.02f}s, AvgLoss {:.06f}".format(epoch, time.time()-time1, epoch_avg_loss))

                state_dict = self.model.module.state_dict()
                os.makedirs(os.path.dirname(saved_model), exist_ok=True)
                torch.save(state_dict, saved_model)
            
            if validloader:
                test_out = self.predict(validloader, True)
                torch.distributed.all_reduce(test_out)
                if self.rank == 0:
                    y_true, y_pred = test_out

            
            torch.cuda.empty_cache()
            if scheduler is not None:
                scheduler.step()


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run Model")
    parser.add_argument("--seq_len",
                        type=int,
                        default=512,
                        help="max sequence length")
    parser.add_argument("--ip",
                        type=str,
                        default="localhost",
                        help="ip address")
    parser.add_argument("--port",
                        type=str,
                        default=str(random.randint(20000, 30000)),
                        help="port num")
    parser.add_argument("--cuda_devices",
                        type=int,
                        nargs='+',
                        default=list(range(torch.cuda.device_count())),
                        help="cuda devices")
    parser.add_argument("--mode",
                        type=str,
                        choices=["train", "eval"],
                        help="train or eval")
    parser.add_argument("--num_worker",
                        type=int,
                        default=8,
                        help="number of data loader worker")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="batch size")
    parser.add_argument("--epoch",
                        type=int,
                        default=5,
                        help="num epoch")
    parser.add_argument("--max_norm",
                        type=int,
                        default=30,
                        help="max norm value")
    return parser.parse_args()

def set_manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def dist_init(ip, rank, local_rank, world_size, port):
    """
        initialize data distributed
    """
    host_addr_full = 'tcp://' + ip + ':' + str(port)
    torch.distributed.init_process_group("nccl", init_method=host_addr_full, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(
                2*hidden_size)] = 1.0

def train_worker(rank, args, world_size):
    model_file = "model.torch"
    device = args.cuda_devices[rank]
    dist_init(args.ip, rank, device, world_size, args.port)
    model = prepare_model(model_file, args, need_load=False, is_train=True, distributed=True)
    criterion = nn.CrossEntropyLoss()

    train_dataset = MyDataset()
    train_datasampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=args.num_worker, batch_size=args.batch_size, sampler=train_datasampler)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=1e-6)

    trainer = Trainer(model, train_dataloader, train_datasampler, device, rank, args)

    valid_dataset = MyDataset()
    valid_datasampler = DistributedSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset,  pin_memory=True, num_workers=args.num_worker, batch_size=args.batch_size, sampler=valid_datasampler)

    trainer.fit(args.epoch, optimizer, criterion, model_file=model_file, scheduler=scheduler,
                validloader=valid_dataloader, validset=valid_dataset)

def prepare_model(model_file, args, need_load=False, is_train=True, distributed=True):
    if distributed:
        rank, device = torch.distributed.get_rank(), torch.cuda.current_device()
    else:
        rank, device = 0, torch.cuda.current_device()
    model = MyModel()
    model = model.to(device)
    if need_load:
        model.load_state_dict(torch.load(model_file, map_location='cuda:{}'.format(device)))
        if rank == 0:
            print("[*] load model {}".format(model_file))
    else:
        model.apply(init_weights)
    if is_train and distributed:
        model = DistributedDataParallel(model, device_ids=[device])
    print("[*] rank:{}, device:{}".format(rank, device))
    return model

def trainer():
    world_size = len(args.cuda_devices)
    mp.spawn(train_worker, args=(args, world_size), nprocs=world_size)


if __name__ == '__main__':
    args = parameter_parser()
    if args.mode == "train":
        trainer()
