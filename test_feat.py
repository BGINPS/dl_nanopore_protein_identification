# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch, os
from dataset.peptidedataset_feat import PeptideDataset_feature, le
from utils import io
from utils.util import load_train_test_dict

from argparse import ArgumentParser
from models.trainer import CNNTrainer


device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args):
    ## generate logdir
    print('Loading test data')

    test_dict = io.read_pickle( args.hp_pkl_file )


    # load data
    test_dataset = PeptideDataset_feature( test_dict, resize_length=args.resize_length, padding=args.padding )

    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    print('Start testing', '-'*20)
    trainer_con = CNNTrainer( None, test_data_loader, args= args, iftest=True )
    trainer_con.model_load( args.checkpoint_file )
    log_dir = os.path.split( args.checkpoint_file )[0]
    cm_csv_file = f'{log_dir}/test_cm.csv'
    fig = trainer_con.plot_cm_for_test_dataset( class_labels=le.classes_, save_cm_to_file = cm_csv_file )
    plt.savefig( f'{log_dir}/test_cm.png' )
    plt.close()
    print( 'Finish testing', '-'*20 )

if __name__ == "__main__":
    parser = ArgumentParser(description='Training peptide classification')
    parser.add_argument('--hp_pkl_file', default='./data/density18k_test.pkl')

    parser.add_argument('--checkpoint_file', default='./logs/model_best.pth')

    parser.add_argument('--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('--resize_length', default=1000, type=int, help="resized peptide length")
    parser.add_argument('--padding', default=0, type=int, help="padding size")

    parser.add_argument('--model_name', default='CNN', choices=['CNN', 'LSTM', 'CNN_STFT', 'LSTM_STFT' ],
                        help='model choice (default is CNN)')

    args = parser.parse_args()

    main( args= args )

