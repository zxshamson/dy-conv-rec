import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, choices=['technology', 'todayilearned', 'funny'])
    parser.add_argument("modelname", type=str, choices=["DCR"])
    parser.add_argument("--cuda_dev", type=str, default="0")
    parser.add_argument("--factor_dim", type=int, default=20)
    parser.add_argument("--neg_sample_num", type=int, default=5)
    parser.add_argument("--kernal_num", type=int, default=100)
    parser.add_argument("--kernal_kind", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mlp_layers_num", type=int, default=2)
    parser.add_argument("--gcn_layers_num", type=int, default=1)
    parser.add_argument("--runtime", type=int, default=0)
    parser.add_argument("--pos_weight", type=float, default=100)
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--att", type=str, default="u", choices=["u", "n"])
    parser.add_argument("--month_num", type=int, default=4)
    parser.add_argument("--pretrained_file", type=str, default='glove.840B.300d.txt')
    parser.add_argument("--no_lstm", action="store_true")
    parser.add_argument("--no_gcn", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_config()

