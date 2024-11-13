import numpy as np

#######################################################
# put `w2_sigmoid_forward` and `w2_sigmoid_grad_input` here #
def w2_sigmoid_forward(x_input):

    output = 1 / (1 + np.exp(-x_input))
    
    return output

def w2_sigmoid_grad_input(x_input, grad_output):

    sigmoid_output = 1 / (1 + np.exp(-x_input))  # Apply sigmoid to get the output
    grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)  # Chain rule
    
    return grad_input

#######################################################


#######################################################
# put `w2_nll_forward` and `w2_nll_grad_input` here    #
def w2_nll_forward(target_pred, target_true):

    output = (- 1 / len(target_true)) * (np.sum((target_true * np.log(target_pred)) + (1 - target_true) * np.log(1 - target_pred)))
    
    return output

def w2_nll_grad_input(target_pred, target_true):

    grad_input = 1 / len(target_pred) * ((target_pred - target_true) / (target_pred * (1 - target_pred)))
    return grad_input
#######################################################
