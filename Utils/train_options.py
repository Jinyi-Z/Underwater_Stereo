import argparse
import os.path as osp
class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive depth estimation netowork")
        parser.add_argument("--model", type=str, default='gwcnet',help="available options: dispnet")
        parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
        parser.add_argument("--source", type=str, default='vkitti',help="source dataset")
        parser.add_argument("--target", type=str, default='kitti',help="target dataset")
        parser.add_argument("--train_batch_size", type=int, default=1, help="train batch size.")
        parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
        parser.add_argument("--num_workers", type=int, default=4, help="number of threads.")
        parser.add_argument("--data_dir_source", type=str, default='/path/to/dataset/source', help="Path to the directory containing the source dataset.")
        parser.add_argument("--data_list_source", type=str, default='/path/to/dataset/source_list', help="Path to the file listing the images in the source dataset.")
        parser.add_argument("--data_dir_target", type=str, default='/path/to/dataset/target', help="Path to the directory containing the target dataset.")
        parser.add_argument("--data_list_target", type=str, default='/path/to/dataset/target_list', help="Path to the file listing the images in the target dataset.")
        parser.add_argument("--data_label_folder_target", type=str, default=None, help="Path to the soft assignments in the target dataset.")
        parser.add_argument("--num-classes", type=int, default=2, help="Number of classes for discriminator.")
        parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="initial learning rate for the deep estimation network.")
        parser.add_argument("--learning_rate_D", type=float, default=1e-4, help="initial learning rate for discriminator.")
        parser.add_argument("--lambda_adv_target", type=float, default=0.001, help="lambda_adv for adversarial training.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--epochs", type=int, default=250000, help="Number of epochs to train.")
        parser.add_argument("--num_steps_stop", type=int, default=120000, help="Number of training steps for early stopping.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
        parser.add_argument("--init_weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore_from", type=str, default=None, help="Where restore model parameters from.")
        parser.add_argument("--save_freq", type=int, default=20, help="Save summaries and checkpoint every often.")
        parser.add_argument("--print_freq", type=int, default=10, help="print loss and time fequency.")
        parser.add_argument("--snapshot_dir", type=str, default='/path/to/snapshots/', help="Where to save snapshots of the model.")
        parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        
        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    
        # save to the disk
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')    
        