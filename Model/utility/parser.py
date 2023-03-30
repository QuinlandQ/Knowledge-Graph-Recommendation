'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    '''
    nargs='*' 　　　表示参数可设置零个或多个
    nargs=' '+' 　　表示参数可设置一个或多个
    nargs='?'　　　表示参数可设置零个或一个
    '''
    parser = argparse.ArgumentParser(description="Run PPKG.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--dataset', nargs='?', default='recipe',
                        help='load dataset from recipe')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=32,
                        help='CF Embedding size.')
    parser.add_argument('--kge_size', type=int, default=32,
                        help='KG Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,32,16]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=1024,
                        help='KG batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-3,1e-1]',
                        help='Regularization for user and item embeddings.')

    parser.add_argument('--n_hidden_units', nargs='?', default=32,
                        help='LSTM hidden units.')
    parser.add_argument('--layer_num', nargs='?', default=2,
                        help='LSTM layer_num.')
    parser.add_argument('--input_size', nargs='?', default=9,
                        help='LSTM input_size.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--model_type', nargs='?', default='ppkg',
                        help='Specify a loss type from {ppkg, bprmf, fm, nfm, cke, cfkg}.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--alg_type', nargs='?', default='bi',
                        help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--node_dropout', nargs='?', default='[0.5]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.5,0.5,0.5]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[10]',
                        help='Output sizes of every layer')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--use_att', type=bool, default=True,
                        help='whether using attention mechanism')
    parser.add_argument('--use_kge', type=bool, default=True,
                        help='whether using knowledge graph embedding')
    parser.add_argument('--loss_type', nargs='?', default=True,
                        help='the model loss type')
    parser.add_argument('--l1_flag', type=bool, default=True,
                        help='Flase: using the L2 norm, True: using the L1 norm.')

    return parser.parse_args()