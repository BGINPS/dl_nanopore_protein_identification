# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import numpy as np
import torch, tqdm
from dataset.peptidedataset_feat import le
from torch.utils.data import TensorDataset, DataLoader
from utils import io
from models import model as md
from utils.util import resize
from argparse import ArgumentParser
from torch import nn

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(args):
    model_name = args.model_name
    output_class = len(le.classes_)
    doc = False

    if model_name.upper() == 'CNN':
        model = md.CNN1D(num_classes=output_class, doc=doc)
    elif model_name.upper() == "RESNET":
        model = md.RESNet1D(num_classes=output_class)
    elif model_name.upper() == "LSTM":
        model = md.CnnLstmNet(output_class)
    elif model_name.upper() == 'CNN_STFT':
        model = md.CNN1D_Stft(num_classes=output_class, doc=doc)

    elif model_name.upper() == 'LSTM_STFT':
        model = md.LSTM_Stft(num_classes=output_class, doc=doc)
    model.to(device)
    return model
def main(args):
    ## generate logdir
    print('Loading inference data')
    inference_dict = io.read_pickle( args.hp_pkl_file )
    keys_list = list( inference_dict.keys() )

    print('Start inference', '-'*20)
    model = get_model( args )
    weights = torch.load( args.checkpoint_file )
    model.load_state_dict(weights)
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    resut_dict = {}
    with torch.no_grad():
        for read_i_key in tqdm.tqdm( keys_list ):
            result_i = {}

            read_i = inference_dict[read_i_key]
            ## cut the peptide signal and normalized by I0
            peptide_signal = read_i['signal'][read_i['window'][0] - args.padding: read_i['window'][1] + args.padding] / \
                             read_i['openpore']
            peptide_signal = resize(peptide_signal, args.resize_length)
            ### todo: make the resize function into transforms
            peptide_signal = torch.tensor(peptide_signal, dtype=torch.float).unsqueeze(0).unsqueeze(0)

            ## collect features: pd/rd
            feature = [read_i['pd2rd'], read_i['window_i2i0_mean'], read_i['window_i2i0_std']]
            peptide_feat = torch.tensor(feature, dtype=torch.float).unsqueeze(0)

            pred_prob = softmax( model(peptide_signal.to(device), peptide_feat.to(device)) ).cpu().detach().numpy()

            pred_class = np.argmax( pred_prob, axis=1 )[0]
            result_i['pred_class'] = pred_class
            result_i['pred_prob'] = pred_prob[0, pred_class]
            resut_dict[read_i_key] = result_i

    io.save_pickle(resut_dict, args.output_pkl_file)


    # del test_dict
    print( 'Finish testing', '-'*20 )

if __name__ == "__main__":
    parser = ArgumentParser(description='inference peptide prediction')
    parser.add_argument('--hp_pkl_file', default='./data/data.pkl')
    parser.add_argument('--output_pkl_file', default='./data/data_inference.pkl')

    parser.add_argument('--checkpoint_file', default='./logs/model_best.pth')

    parser.add_argument('--batch_size', default=64, type=int, help="batch size")
    parser.add_argument('--resize_length', default=1000, type=int, help="resized peptide length")
    parser.add_argument('--padding', default=100, type=int, help="padding size")
    parser.add_argument('--model_name', default='CNN', choices=['CNN', 'LSTM', 'CNN_STFT', 'LSTM_STFT' ],
                        help='model choice (default is CNN)')

    args = parser.parse_args()
    main( args= args )

