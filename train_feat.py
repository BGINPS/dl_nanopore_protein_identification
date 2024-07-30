# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from torch.utils.data import TensorDataset, DataLoader
import torch, os
from dataset.peptidedataset_feat import PeptideDataset_feature
from datetime import datetime
from utils import io
from utils.util import load_train_test_dict, get_logger, seed_everything

from argparse import ArgumentParser
from models.trainer import CNNTrainer


device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args):

    seed_everything()
    ## generate logdir
    today = datetime.today().strftime('%y%m%d')
    exp_name = f'{args.exp_name}_{today}_lr{args.lr}_bs{args.batch_size}_Ps{args.resize_length}_N{args.Num_epoch}_Pad{args.padding}'
    args.logdir = f'{args.logdir}/{exp_name}/'
    os.makedirs(args.logdir, exist_ok=True)
    logger = get_logger(args.logdir)
    logger.info( args )
    logger.info('Loading data for training and validataion')

    # load train and validation data
    train_dict = io.read_pickle( args.train_pkl_file )
    val_dict = io.read_pickle(args.val_pkl_file)

    train_dataset = PeptideDataset_feature( train_dict, resize_length= args.resize_length, padding=args.padding )
    val_dataset = PeptideDataset_feature( val_dict, resize_length=args.resize_length, padding=args.padding )

    num_train_instances, num_val_instances = len(train_dataset), len(val_dataset)
    logger.info( f'Number of training: {num_train_instances} \t number of validation: {num_val_instances}' )
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    logger.info('Start training')
    trainer_con = CNNTrainer(train_data_loader, val_data_loader, args= args, logger=logger,
                             eva_test_dataset_each_epoch = True, device='gpu', alpha=3.0 )

    trainer_con.train()

if __name__ == '__main__':
    parser = ArgumentParser(description='Training peptide classification')
    parser.add_argument('--train_pkl_file', default='./data/density_train.pkl')
    parser.add_argument('--val_pkl_file', default='./data/density_val.pkl')

    parser.add_argument('--exp_name', default='Peptide_clean_density', type=str)
    parser.add_argument('--logdir', default='./logs')
    parser.add_argument('--Num_epoch', default=100, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    ## hyperparameters
    parser.add_argument('--lr', default=0.005, type=float, help="batch size")
    parser.add_argument('--batch_size', default=512, type=int, help="batch size")
    parser.add_argument('--resize_length', default=1000, type=int, help="resized peptide length")
    parser.add_argument('--padding', default=0, type=int, help="padding size")

    parser.add_argument('--model_name', default='CNN', choices=['CNN','RESNET', 'LSTM', 'CNN_STFT', 'LSTM_STFT' ],
                        help='model choice (default is CNN)')

    args = parser.parse_args()
    args.exp_name = args.exp_name + '_model-' + args.model_name
    # print("Start training: ", args.exp_name)
    main( args= args)

