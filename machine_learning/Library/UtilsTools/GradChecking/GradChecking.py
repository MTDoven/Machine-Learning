import numpy as np


def estimate_grad(function:callable, param_left:tuple, param_right:tuple, dx:float)->tuple:
    """
    Used in grad check to estimate the grad
    If param_left is param_right, this function will try to change the param in the 'function'
    :param function: Input a function you want to check grad
    :param param_left: The left_X(first) you want to input
    :param param_right: The right_X(second) you want to input
    :param dx: The dx (have NOT time 2) or the bias you used in params
    :return: grad_approx: estimated grad; grad_calculated: Grad return by the function
    """
    if param_left is param_right:
        assert hasattr(function, 'W'), \
            "Cannot get function.W. check if 'param_left'=='param_right' or check function.W"
        function.W[-1] -= dx
        result1 = function(*param_left)
        function.W[-1] += 2*dx
        result2 = function(*param_right)
        function.W[-1] -= dx
        grad_approx = (result2-result1)/(2*dx)
        grad_calculated = function.get_grad_W()
    else:
        grad_approx = (function(*param_right)-function(*param_left))/(2*dx)
        grad_calculated = function.get_grad()
    return grad_approx, grad_calculated


