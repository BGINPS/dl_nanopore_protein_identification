import torch
from torch import nn
import torch.optim as optim
from . import model as md
import numpy as np
import pandas as pd
# from models.resnet1D import ResNet1D
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dataset.peptidedataset_feat import le
from tqdm import tqdm




def doc_loss(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    eps: float=1e-5
) -> torch.Tensor:
    """Calculate doc loss

    Args:
        preds (torch.Tensor): N * M tensor. N: sample number, M: class number. all elements are in 0-1
        targets (torch.Tensor): the same shape with preds, all elements are either 0 or 1, indicating class
        eps (float, optional): eps. Defaults to 1e-10.

    Returns:
        torch.Tensor: doc loss
    """
    # DOC: Deep Open Classification of Text Documents
    p = preds
    min_c = torch.Tensor([eps])
    max_c = torch.Tensor([1-eps])
    if p.is_cuda:
        min_c = min_c.to(torch.device('cuda:0'))
        max_c = max_c.to(torch.device('cuda:0'))
        
    p = torch.max(p, min_c)
    p = torch.min(p, max_c)
    loss = torch.sum(-torch.log(p)*targets) + torch.sum(-torch.log(1-p)*(1-targets))
    return loss

class CNNTrainer():
    def __init__(self, train_dataloader, test_dataloader, args, logger=None, eva_test_dataset_each_epoch:bool = True,
                 device='gpu', doc:bool=False, iftest:bool=False, alpha=3.0):
        self.args = args
        model = args.model_name
        self.doc = doc

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.eva_test_dataset_each_epoch = eva_test_dataset_each_epoch
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and device.upper()=='GPU' else 'cpu')

        self.output_class = len(le.classes_)
        self.alpha = alpha
        if model.upper() == 'CNN':
            self.model = md.CNN1D(num_classes=self.output_class, doc=self.doc)
        elif model.upper() == "RESNET":
            self.model = md.RESNet1D(num_classes=self.output_class)
        elif model.upper() == "LSTM":
            self.model = md.CnnLstmNet(self.output_class)
        elif model.upper() == 'CNN_STFT':
            self.model = md.CNN1D_Stft(num_classes=self.output_class, doc=self.doc)

        elif model.upper() == 'LSTM_STFT':
            self.model = md.LSTM_Stft(num_classes=self.output_class, doc=self.doc)
        self.model.to(self.device)
        self.doc_thresholds = (torch.ones([self.output_class]) * 0.5).to(self.device)

        if not iftest:
            self.lr = args.lr
            self.epoch = args.Num_epoch
            self.logger = logger

            self.loss_fun = doc_loss if self.doc else nn.CrossEntropyLoss(reduction='sum')
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                    min_lr=0.01*self.lr,  patience=10, verbose=True)

            self.logger.info(f'Model {model} has total parameter number: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.2f} M')
        
    def model_load(self, checkpt):
        weights = torch.load( checkpt )
        self.model.load_state_dict( weights )
    def train(self,):
        ## prepare tensorboard summary writer
        swriter = SummaryWriter(log_dir=self.args.logdir)

        best_acc = 0
        for epoch in range(self.epoch):
            # train  model
            self.model.train()
            losses_in_an_epoch, accs_num_in_an_epoch, sample_num_in_an_epoch, all_outputs_for_an_epoch, all_ys_for_an_epoch = [], [], [], [], []
            for indx, ( X,f, y) in enumerate( tqdm( self.train_dataloader, desc=f'Train: {epoch} / {self.epoch}' ) ):
                X, y = X.to(self.device), y.to(self.device)
                f = f.to(self.device)
                outputs = self.model(X, f)
                all_outputs_for_an_epoch.append(outputs)
                all_ys_for_an_epoch.append(y)
                loss = self.loss_fun(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_in_an_epoch.append(loss.item())
                accs_num_in_an_epoch.append(torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(y, dim=1)).item())
                sample_num_in_an_epoch.append(len(X))

            if epoch % 10 ==0:
                all_outputs_for_an_epoch = torch.cat(all_outputs_for_an_epoch, dim=0)
                all_ys_for_an_epoch = torch.cat(all_ys_for_an_epoch, dim=0)
                if self.doc:
                    self.update_doc_thresholds(all_outputs_for_an_epoch, all_ys_for_an_epoch)
                    print(self.doc_thresholds)

            training_loss = np.sum(losses_in_an_epoch) / np.sum(sample_num_in_an_epoch)
            training_acc = np.sum(accs_num_in_an_epoch) / np.sum(sample_num_in_an_epoch)

            if not self.eva_test_dataset_each_epoch:
                print('\n', end='')
                continue

            # Validation
            self.model.eval()
            test_losses_in_an_epoch, test_accs_num_in_an_epoch, test_sample_num_in_an_epoch = [], [], []
            with torch.no_grad():
                for indx, ( X, f, y) in enumerate( tqdm(self.test_dataloader, desc=f'Validation: {epoch} / {self.epoch}') ):
                    X, y = X.to(self.device), y.to(self.device)
                    f = f.to(self.device)
                    outputs = self.model(X, f)
                    if self.doc:
                        loss = self.loss_fun(outputs[y[:,-1]==0], y[y[:,-1]==0,0:-1])
                    else:
                        loss = self.loss_fun(outputs, y)
                    test_losses_in_an_epoch.append(loss.item())
                    test_accs_num_in_an_epoch.append(torch.sum(self.output_to_class(outputs) == torch.argmax(y, dim=1)).item())
                    test_sample_num_in_an_epoch.append(len(X))
            test_loss = np.sum(test_losses_in_an_epoch) / np.sum(test_sample_num_in_an_epoch)
            test_acc = np.sum(test_accs_num_in_an_epoch) / np.sum(test_sample_num_in_an_epoch)
            self.logger.info(f'epoch: {epoch + 1}, training loss: {training_loss:.8f} training acc: {training_acc:.4f}, '
                             f'validation loss: {test_loss:.8f}, validation acc: {test_acc:.4f}')

            self.scheduler.step( test_loss )
            swriter.add_scalar('loss/train', training_loss, epoch)
            swriter.add_scalar('loss/validation', test_loss, epoch)
            swriter.add_scalar('acc/train', training_acc, epoch)
            swriter.add_scalar('acc/validation', test_acc, epoch)
            if best_acc < test_acc:
                best_acc = test_acc

                save_path = f'{self.args.logdir}/model_best.pth'
                csv_file = f'{self.args.logdir}/val_cm.csv'

                torch.save(self.model.state_dict(), save_path)

                fig = self.plot_cm_for_test_dataset( le.classes_, save_cm_to_file=csv_file )
                swriter.add_figure('test_cm', fig, global_step= epoch)
                plt.close(fig)
        swriter.close()


    def update_doc_thresholds(self, all_outputs_for_an_epoch, all_ys_for_an_epoch):
        for c in range(all_ys_for_an_epoch.shape[1]):
            probs = all_outputs_for_an_epoch[all_ys_for_an_epoch[:,c]==1,c]
            pseudo_probs = torch.Tensor([1-i+1 for i in probs]).to(self.device)
            all_probs = torch.cat([probs, pseudo_probs])
            self.doc_thresholds[c] = max(0.5, 1 - all_probs.std() * self.alpha)


    def output_to_class(self, outputs):
        class_labels = torch.argmax(outputs, dim=1)
        if self.doc:
            lower_than_cut = torch.zeros_like(outputs)
            lower_than_cut[outputs < self.doc_thresholds] = 1
            class_labels[torch.sum(lower_than_cut, dim=1) == self.output_class] = self.output_class
        return class_labels


    def predict_test_dataset(self,):
        return self.predict_dataloader(self.test_dataloader)
    
    def predict_test_dataset_with_label(self, classes):
        read_ids, preds, trues = self.predict_dataloader(self.test_dataloader)
        df = pd.DataFrame({'read_id': read_ids, 'pred': preds, 'true': trues})
        c = np.array(classes)
        df['pred'] = c[df['pred']]
        df['true'] = c[df['true']]
        df = df.set_index('read_id')
        return df

    def predict_train_dataset(self,):
        return self.predict_dataloader(self.train_dataloader)

    def predict_dataloader(self, dataloader):
        self.model.eval()
        all_outputs, ys, all_read_ids = [], [], []
        with torch.no_grad():
            for indx, (X,f, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                f = f.to(self.device)
                all_outputs.append(self.model(X,f))
                ys.append(y)
        all_preds = self.output_to_class(torch.cat(all_outputs, dim=0)).cpu().numpy()
        all_ys = torch.argmax(torch.cat(ys, dim=0), dim=1).cpu().numpy()
        return all_read_ids, all_preds, all_ys
    

    def get_cm(self, true, pred, class_labels, fill_empty_class=True):
        cm_df = pd.DataFrame({'pred': pred, 'true': true})
        cm_df['count'] = 1
        cm_df = cm_df.groupby(["pred", 'true']).sum().reset_index()
        if fill_empty_class:
            for i in np.setdiff1d(np.unique(true), np.unique(pred)):
                cm_df.loc[len(cm_df)] = [i,0,0]
            for j in np.setdiff1d(np.unique(pred),np.unique(true)):
                cm_df.loc[len(cm_df)] = [0,j,0]
        cm_df = cm_df.pivot(index='true', columns='pred', values='count').fillna(0)
        cm_df.columns = class_labels
        cm_df.index = class_labels
        cm_df.columns.name = 'pred'
        cm_df.index.name = 'true'
        return cm_df


    def plot_cm_for_test_dataset(
        self,
        class_labels: list,
        nor_to_percent_for_each_pred: bool = True,
        mark_diagonal_line: bool = True,
        lw_of_rectangle: int = 1,
        annot: bool = True,
        annot_size=8,
        figsize=(14,10),
        save_cm_to_file=None,
    ):
        
        _, pred, true = self.predict_test_dataset()
        return self.cal_cm_and_plot(class_labels, nor_to_percent_for_each_pred, mark_diagonal_line, lw_of_rectangle, annot, annot_size, pred, true, 
                                    save_cm_to_file=save_cm_to_file, figsize=figsize)
    

    def plot_cm_for_train_dataset(
        self,
        class_labels: list,
        nor_to_percent_for_each_pred: bool = True,
        mark_diagonal_line: bool = True,
        lw_of_rectangle: int = 1,
        annot: bool = True,
        annot_size=8,
        figsize=(10,10)
    ):
        if self.doc:
            self.doc = False
            _, pred, true = self.predict_train_dataset()
            self.doc = True
        else:
            _, pred, true = self.predict_train_dataset()
        ax = self.cal_cm_and_plot(class_labels, nor_to_percent_for_each_pred, mark_diagonal_line, lw_of_rectangle, annot, annot_size, pred, true, figsize=figsize)
        return ax

    def cal_cm_and_plot(self, class_labels, nor_to_percent_for_each_pred, mark_diagonal_line, lw_of_rectangle, annot, annot_size, pred, true, save_cm_to_file=None, figsize=(10,10)):
        cm_df = self.get_cm(true, pred, class_labels)
        if save_cm_to_file:
            cm_df.to_csv(save_cm_to_file)
        fig = self.plot_cm(cm_df=cm_df,
                          nor_to_percent_for_each_pred=nor_to_percent_for_each_pred,
                          figsize=figsize, 
                          annot=annot,
                          annot_size=annot_size,
                          lw_of_rectangle=lw_of_rectangle,
                          mark_diagonal_line=mark_diagonal_line)
        return fig
                

    def read_csv_of_cm_and_plot(
        self, 
        csv_file_of_cm: str,
        nor_to_percent_for_each_pred: bool = True, 
        figsize: tuple = (10,10), 
        annot: bool = True,
        annot_size: float = 8.0,
        lw_of_rectangle: float = 1.0,
        mark_diagonal_line: bool = True
    ):
        cm_df = pd.read_csv(csv_file_of_cm, index_col=0)
        cm_df.columns.name = 'pred'
        cm_df.index.name = 'true'
        fig = self.plot_cm(cm_df=cm_df,
                     nor_to_percent_for_each_pred=nor_to_percent_for_each_pred,
                     figsize=figsize,
                     annot=annot,
                     annot_size=annot_size,
                     lw_of_rectangle=lw_of_rectangle,
                     mark_diagonal_line=mark_diagonal_line)
        return fig

    def plot_cm(
        self,
        cm_df,
        nor_to_percent_for_each_pred: bool = True, 
        figsize: tuple = (10,10), 
        annot: bool = True,
        annot_size: float = 8.0,
        lw_of_rectangle: float = 1.0,
        mark_diagonal_line: bool = True
    ):
        if nor_to_percent_for_each_pred:
            cm_df = cm_df.div(cm_df.sum(axis=1), axis='rows')*100
        fig, ax = plt.subplots(figsize=figsize)
        cmap = sns.cubehelix_palette(20, light=0.95, dark=0.15)
        sns.heatmap(cm_df, annot=annot, annot_kws={"size": annot_size}, cmap=cmap, fmt=".1f", ax=ax, vmin=0, vmax=100)

        if mark_diagonal_line:
            for i in range(len(cm_df)):
                ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=lw_of_rectangle, clip_on=False))
        plt.yticks(rotation=0)
        plt.xticks(rotation=45)
        return fig


                




            
class VAETrainer():
    def __init__(self, train_dataloader, test_dataloader, eva_test_dataset_each_epoch:bool = True, device='gpu', lr=0.05, epoch=200, doc:bool=False, alpha=3.0,):
        self.lr = lr
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.eva_test_dataset_each_epoch = eva_test_dataset_each_epoch
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and device.upper()=='GPU' else 'cpu')

        self.output_class = len(self.train_dataloader.dataset.y[0])
        
        self.vae_model = md.VAE(input_dim=1000, latent_dim=[256, 128, 64, 50])
        self.cls_model = md.Classifier(input_dim=1000, latent_dim=[1024, 1024, 1024, 1024, 1024], output_dim=self.output_class)

        self.recon_loss_fun = nn.MSELoss(reduction='mean')
        self.kl_loss_fun = lambda mean, log_var: -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        self.classification_loss_fun = nn.CrossEntropyLoss(reduction='sum')

        self.optimizer = optim.Adam(self.vae_model.parameters(), lr=self.lr)
        self.optimizer_cls = optim.Adam(self.cls_model.parameters(), lr=self.lr)

        self.vae_model.to(self.device)
        self.cls_model.to(self.device)

        print(f'total parameter number of vae model: {sum(p.numel() for p in self.vae_model.parameters() if p.requires_grad)}')
        print(f'total parameter number of cls model: {sum(p.numel() for p in self.cls_model.parameters() if p.requires_grad)}')
        
    
    def train_vae(self,):
        for epoch in range(self.epoch):
            # train loss
            self.vae_model.train()
            losses_in_an_epoch, sample_num_in_an_epoch = [], []
            for indx, (_, X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                z, recon_x, mean, log_var = self.vae_model(X)

                loss = self.recon_loss_fun(recon_x, X) + self.kl_loss_fun(mean, log_var)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_in_an_epoch.append(loss.item())
                sample_num_in_an_epoch.append(len(X))
            
            print(f'epoch: {epoch + 1:5d}, training loss: {np.mean(losses_in_an_epoch):.8f}')
        
    def train_cls(self, ):
        self.vae_model.eval()
        for epoch in range(self.epoch):
            self.cls_model.train()
            losses_in_an_epoch, accs_num_in_an_epoch, sample_num_in_an_epoch, all_outputs_for_an_epoch, all_ys_for_an_epoch = [], [], [], [], []
            for indx, (_, X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.cls_model(X, self.vae_model)
                outputs.squeeze_(1)
                all_outputs_for_an_epoch.append(outputs)
                all_ys_for_an_epoch.append(y)
                loss = self.classification_loss_fun(outputs, y)


                self.optimizer_cls.zero_grad()
                loss.backward()
                # self.optimizer.step()
                self.optimizer_cls.step()
                losses_in_an_epoch.append(loss.item())
                accs_num_in_an_epoch.append(torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(y, dim=1)).item())
                sample_num_in_an_epoch.append(len(X))

            print(f'epoch: {epoch + 1:5d}, training loss: {np.sum(losses_in_an_epoch) / np.sum(sample_num_in_an_epoch):.8f} \
                  training acc: {np.sum(accs_num_in_an_epoch) / np.sum(sample_num_in_an_epoch):.4f}', end="\n")



if __name__ == "__main__":
    CNNTrainer