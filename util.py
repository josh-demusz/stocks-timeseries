import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_multi_curves(x, y_list, x_axis, y_axis, y_labels, title, legend_title=None):
    fig, ax1 = plt.subplots(1, 1)

    # ax1.set_ylim([0, 0.7])

    color_pool = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lime', 'steelblue', 'maroon', 'orange']

    for i, y in enumerate(y_list):
        if len(y) < len(x):
            x_temp = x[:len(y)]
        else:
            x_temp = x

        if len(x) < 30:
            ax1.plot(x_temp, y, marker='o', color=color_pool[i], label=y_labels[i])
        else:
            ax1.plot(x_temp, y, color=color_pool[i], label=y_labels[i])
    ax1.set_xlabel(x_axis)
    ax1.set_ylabel(y_axis)
    ax1.set_title(title, pad=20)
    # ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 0.5))
    if legend_title == None:
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=legend_title)

    fig.subplots_adjust(bottom=0.1)
    plt.tight_layout()

    filename = title.lower()
    filename = filename.replace(" ", "-")

    # plt.savefig("{}.png".format(filename), bbox_inches='tight',
    #             pad_inches=.1)
    #
    # plt.clf()
    # plt.cla()
    # plt.close()
    plt.show()
    return

# Citation: https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
def interleave(x, size):
    s = list(x.shape)
#     print('s: {}'.format(s))
#     print('x.reshape([-1, size] + s[1:]): {}'.format(x.reshape([-1, size] + s[1:]).shape))
#     print('.transpose(0, 1): {}'.format(x.reshape([-1, size] + s[1:]).transpose(0, 1).shape))
#     print('.reshape([-1] + s[1:]): {}'.format(x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:]).shape))
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# model = models.build_wideresnet(depth=args.model_depth,
#                                             widen_factor=args.model_width,
#                                             dropout=0,
#                                             num_classes=args.num_classes)

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# From: https://towardsdatascience.com/dataloader-for-sequential-data-using-pytorch-deep-learning-framework-part-2-ed3ad5f6ad82
def collate_fn(data):
	'''
	We should build a custom collate_fn rather than using default collate_fn,
	as the size of every sentence is different and merging sequences (including padding)
	is not supported in default.
	Args:
		data: list of tuple (training sequence, label)
	Return:
		padded_seq - Padded Sequence, tensor of shape (batch_size, padded_length)
		length - Original length of each sequence(without padding), tensor of shape(batch_size)
		label - tensor of shape (batch_size)
    '''

    #sorting is important for usage pack padded sequence (used in model). It should be in decreasing order.
	data.sort(key=lambda x: len(x[0]), reverse=True)
	sequences, label = zip(*data)
	length = [len(seq) for seq in sequences]
	padded_seq = torch.zeros(len(sequences), max(length))
	for i, seq in enumerate(sequences):
		end = length[i]
		padded_seq[i,:end] = seq
	return padded_seq, torch.from_numpy(np.array(length)), torch.from_numpy(np.array(label))
