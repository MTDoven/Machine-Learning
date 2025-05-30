import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import vgg
import thop


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--dir_data', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=11,
                    help='depth of the vgg')
parser.add_argument('--baseline', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--pruned', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dir_data, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.dir_data, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


# Load the baseline VGG11 model
model = vgg.VGG(dataset=args.dataset, depth=args.depth)
print("\n\n\n=> loading checkpoint '{}'".format(args.baseline))
checkpoint = torch.load(args.baseline, weights_only=False)
model.load_state_dict(checkpoint['state_dict'], strict=False)

if args.cuda:
    model.cuda()
# 测试剪枝前模型精度
acc = test(model)
# 初始化一个数据输入模型，计算剪枝前模型的参数量与计算量
x = torch.randn(1, 3, 32, 32)
flops, params = thop.profile(model, inputs=(x,))
print("before prune:")
print("params:", params)
print("FLOPs:", flops)


# 加载剪枝后微调前的模型
cfg = [64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256]
new_model_pruned = vgg.VGG(dataset=args.dataset, depth=args.depth, cfg=cfg)
print("\n\n\n=> loading checkpoint '{}'".format(args.pruned))
checkpoint = torch.load(args.pruned, weights_only=False)
new_model_pruned.load_state_dict(checkpoint['state_dict'], strict=False)

if args.cuda:
    new_model_pruned.cuda()

# 测试剪枝后微调前的模型精度
acc = test(new_model_pruned)

# 同样初始化一个数据输入模型，计算剪枝后微调前模型的参数量与计算量
x = torch.randn(1, 3, 32, 32)
flops, params = thop.profile(new_model_pruned, inputs=(x,))
print("after prune, before finetune:")
print("params:", params)
print("FLOPs:", flops)


# Load the fine-tuned model
new_model_finetune = vgg.VGG(dataset=args.dataset, depth=args.depth, cfg=cfg)
print("\n\n\n=> loading checkpoint '{}'".format(args.finetune))
checkpoint = torch.load(args.finetune, weights_only=False)
new_model_finetune.load_state_dict(checkpoint['state_dict'], strict=False)

if args.cuda:
    new_model_finetune.cuda()

# 测试微调后模型精度
acc = test(new_model_finetune)

# 同样初始化一个数据输入模型，计算微调后模型的参数量与计算量
x = torch.randn(1, 3, 32, 32)
flops, params = thop.profile(new_model_finetune, inputs=(x,))
print("after finetune:")
print("params:", params)
print("FLOPs:", flops)
