import os
import argparse

parser = argparse.ArgumentParser("ana")
parser.add_argument('filepath', type=str,
                    default='./stats-cifar-wideResNet/tct/tct_joint_l2_05+05_pgd-7_10.txt')
args = parser.parse_args()

if __name__ == "__main__":
    filename = args.filepath
    triplets = list()
    with open(filename, "r") as f:
        line = f.readline().rstrip()
        while line:
            ind, clean_acc, adv_acc = line.split(' ')
            clean_acc = float(clean_acc)
            adv_acc = float(adv_acc)
            ind = int(ind)
            triplets.append((ind, clean_acc, adv_acc))
            line = f.readline().rstrip()
    triplets.sort(key=lambda t: (t[2], t[1]), reverse=True)
    for i in range(10):
        print("{}: {:4f} {:4f}".format(ind, clean_acc*100, adv_acc*100))
