import numpy as np
import torch
from torch import Tensor
from typing import Iterable
from scipy.optimize import linear_sum_assignment


def assignment_algorithm(cost_matrix, number=10):
    cost_matrix = cost_matrix.cpu().numpy()
    results = []
    for i in range(number):
        cost_matrix = cost_matrix.copy()
        np.random.shuffle(cost_matrix)
        col, row = linear_sum_assignment(cost_matrix.T)
        results.append((row, cost_matrix[row, col].sum()))
    results.sort(key=lambda x: x[1])
    return results


def normalize(tensor: Tensor):
    return (tensor - tensor.mean()) / tensor.std()


def transform(tensor: Tensor, dim: int, indices: Iterable):
    tensor.transpose_(dim, -1)
    tensor = tensor[..., indices]
    tensor.transpose_(dim, -1)
    return tensor


def solve_sequential(source_list: list, target_list: list):
    prev = list(range(source_list[0].in_features))
    beams = [(source_list, prev, 0.),]
    for index in range(len(source_list)-1):
        new_beams = []
        for this_source_list, prev, this_score in beams:
            this_source_list[index] = this_source_list[index].transform_fm_prev(prev)
            solutions = this_source_list[index].search_transforms(target_list[index])
            for (solution, next_score) in solutions:
                new_source_list = this_source_list.copy()
                new_source_list[index] = new_source_list[index].transform_to_next(solution)
                new_beams.append((new_source_list, solution, this_score + next_score))
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:source_list[0].beam_width]
    # the last one
    new_beams = []
    for this_source_list, prev, this_score in beams:
        this_source_list[index+1] = this_source_list[index+1].transform_fm_prev(prev)
        score = this_source_list[index+1].compare_with(target_list[index+1])
        solution = list(range(source_list[-1].out_features))
        new_beams.append((this_source_list, solution, this_score + score))
    new_beams.sort(key=lambda x: x[2])
    beams = new_beams
    # output
    this_source_list, prev, this_score = beams[0]
    return this_source_list




class Module:
    search_number = 1000
    beam_width = 100


class Linear(Module):
    def __init__(self, name: str, weight: Tensor, bias: Tensor):
        self.name = name
        self.weight = weight.clone()
        self.bias = bias.clone()
        self.in_features = weight.size(1)
        self.out_features = weight.size(0)
        self.input_dim = 1
        self.output_dim = 0

    def transform_fm_prev_(self, indices: Iterable):
        assert self.weight.size(self.input_dim) == len(indices), \
            f"{self.weight.size(self.input_dim)} != {len(indices)}"
        self.weight = transform(self.weight, self.input_dim, indices)
        return self

    def transform_to_next_(self, indices: Iterable):
        assert self.weight.size(self.output_dim) == len(indices), \
            f"{self.weight.size(self.output_dim)} != {len(indices)}"
        self.weight = transform(self.weight, self.output_dim, indices)
        self.bias = transform(self.bias, self.output_dim, indices)
        return self

    def transform_fm_prev(self, indices: Iterable):
        module = Linear(name=self.name, weight=self.weight, bias=self.bias)
        module.transform_fm_prev_(indices)
        return module

    def transform_to_next(self, indices: Iterable):
        module = Linear(name=self.name, weight=self.weight, bias=self.bias)
        module.transform_to_next_(indices)
        return module

    def search_transforms(self, target_module: "Linear"):
        target_weight, target_bias = target_module.weight, target_module.bias
        target_weight, target_bias = normalize(target_weight), normalize(target_bias)
        target = torch.cat((target_weight, target_bias.unsqueeze(1)), dim=1)
        assert self.weight.dim() == 2 and target.dim() == 2
        weight, bias = normalize(self.weight), normalize(self.bias)
        source = torch.cat((weight, bias.unsqueeze(1)), dim=1)
        route_matrix = torch.cdist(source, target, p=2)
        solutions = assignment_algorithm(cost_matrix=route_matrix, number=self.search_number)
        solutions = solutions[:self.beam_width]
        return solutions

    def compare_with(self, target_module: "Linear"):
        target_weight, target_bias = target_module.weight, target_module.bias
        target = torch.cat((target_weight, target_bias.unsqueeze(1)), dim=1)
        assert self.weight.dim() == 2 and target.dim() == 2
        source = torch.cat((self.weight, self.bias.unsqueeze(1)), dim=1)
        route_matrix = torch.cdist(source, target, p=2)
        return torch.trace(route_matrix)

    def normalize(self, tensor):
        return (tensor - tensor.mean()) / tensor.std()

    @property
    def dict(self):
        return {self.name+".weight": self.weight.clone(), self.name+".bias": self.bias.clone()}

    def __repr__(self):
        return f"Linear:\nweight: {self.weight}\nbias: {self.bias}\n"

    def __str__(self):
        return self.__repr__()




if __name__ == "__main__":
    def diction_to_list(diction):
        linear1 = Linear("linear1", diction['linear1.weight'], diction['linear1.bias'])
        linear2 = Linear("linear2", diction['linear2.weight'], diction['linear2.bias'])
        linear3 = Linear("linear3", diction['linear3.weight'], diction['linear3.bias'])
        return [linear1, linear2, linear3]
    def list_to_diction(module_list):
        diction = {}
        diction.update(module_list[0].dict)
        diction.update(module_list[1].dict)
        diction.update(module_list[2].dict)
        return diction
    ckpt1 = torch.load("./model_ckpt1.pth", weights_only=True)
    ckpt2 = torch.load("./model_ckpt2.pth", weights_only=True)

    # print original error
    error = 0
    for (k1, v1), (k2, v2) in zip(ckpt1.items(), ckpt2.items()):
        error = error + ((normalize(v1) - normalize(v2)) ** 2).mean()
    print(error)

    # solve sequential
    after_ckpt1 = solve_sequential(source_list=diction_to_list(ckpt1), target_list=diction_to_list(ckpt2))
    after_ckpt1 = list_to_diction(after_ckpt1)
    torch.save(after_ckpt1, "test_ckpt.pth")

    # print after error
    error = 0
    for (k1, v1), (k2, v2) in zip(after_ckpt1.items(), ckpt2.items()):
        error = error + ((normalize(v1) - normalize(v2)) ** 2).mean()
    print(error)
