import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext


def optimal_est(P, order, sequence_length, generator, extra_args):
    x, y = get_batch(P, order, sequence_length, 1024, generator, extra_args)
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(P.device)
    opt_logits = torch.zeros(x.size(0), x.size(1), P.size(1), device=P.device)
    if order > 1:
        opt_logits[:,:order-1,:] = 0.5*torch.ones(x.size(0), order-1, P.size(1), device=P.device)
    for i in range(order-1, x.size(1)):
        idx = x[:,i-order+1:i+1].float() @ powers
        opt_logits[:,i,:] = P[idx.to(int)]
    opt_logits = torch.log(opt_logits)
    opt_loss = F.nll_loss(opt_logits.view(-1, opt_logits.size(-1)), y.view(-1), ignore_index=-1)

    return opt_loss


def get_batch(P, order, seq_length, batch_size, generator, extra_args):
    data = torch.zeros(batch_size, seq_length+1, device=extra_args.device)
    if extra_args.initial == 'steady':
        if P.size(0) == 2:
            alpha = P[1,0] / (P[0,1] + P[1,0])
        else:
            alpha = 0.5
    elif extra_args.initial == 'uniform':
        alpha = 0.5
    else:
        alpha = 0.5
    # Generate first k bits
    for k in range(order):
        data[:,k] = torch.bernoulli(alpha*torch.ones((batch_size,), device=extra_args.device), generator=generator)
    for i in range(order, seq_length):
        data[:,i] = get_next_symbols(P, order, data[:,i-order:i])
    x = data[:,:seq_length].to(int)
    y = data[:,1:].to(int)
    
    return x, y

def get_next_symbols(P, order, data):
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(data.device)
    idx = data @ powers
    M = P[idx.to(int)]
    s = torch.multinomial(M,1).flatten()

    return s


@torch.no_grad()
def eval(model, P, order, sequence_length, batch_size, generator, extra_args, max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(P, order, sequence_length, batch_size, generator, extra_args)
        with ctx:
            outputs = model(x, targets=y)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


@torch.no_grad()
def eval_probs(model, P, order, sequence_length, generator, extra_args, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    x, y = get_batch(P, order, sequence_length, 1, generator, extra_args)
    with ctx:
        outputs = model (x, targets=y)
    val_loss = outputs['loss']
    loss_list_val.append(val_loss)
    acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    probs = F.softmax(outputs['logits'], dim=-1)

    xb = x[0].float()
    probsb = probs[0, order-1:]
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(extra_args.device)
    idx = torch.Tensor([xb[i:i+order] @ powers for i in range(sequence_length - order + 1)])
    prob_vec = []
    for i in range(2**order):
        vec = probsb[idx == i][:,1] # estimated p
        prob_vec.append(vec)

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    opt_loss = optimal_est(P, order, sequence_length, generator, extra_args)

    return val_acc, val_loss, val_perplexity, prob_vec, opt_loss


def save_checkpoint(model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
