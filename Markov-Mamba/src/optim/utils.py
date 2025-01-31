from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from contextlib import nullcontext


def get_random_P(order, batch_size, generator, device, dtype):
    pk = torch.rand((batch_size, 2**order, 1), generator=generator, dtype=dtype, device=device)
    P = torch.cat([1 - pk, pk], dim=2)

    return P

def empirical_est(x, y, type, order, window=0, beta=1, save_counts=False):
    assert x.size(0) == 1
    assert beta > 0

    seq_length = x.size(1)
    device = x.device
    x = x.float().squeeze()
    y = y.float().squeeze()
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(device)
    idx = F.conv1d(x.view(1, -1), powers.view(1, 1, -1)).squeeze()
    est_vec = []
    if type == "jump-markov":
        jump = seq_length // 2
        if window in range(1, jump): # Windowed add-beta estimator
            # First half of sequence
            idx1 = idx[:jump-order+1]
            y1 = y[order-1:jump]
            for i in range(2**order):
                mask = (idx1 == i)
                s = torch.stack((y1[order-1:] * mask.int(), mask.int()))
                s = F.pad(s, (window, 0))[:,:-1]
                s = F.conv1d(s, torch.ones(2, 1, window, device=device), groups=2)
                p = (s[0] + beta) / (s[1] + 2*beta)
                est_vec.append(p[mask])
            # Second half of sequence
            idx2 = idx[jump-order+1:]
            y2 = y[jump:]
            for i in range(2**order):
                mask = (idx2 == i)
                s = torch.stack((y2[order-1:] * mask.int(), mask.int()))
                s = F.pad(s, (window, 0))[:,:-1]
                s = F.conv1d(s, torch.ones(2, 1, window, device=device), groups=2)
                p = (s[0] + beta) / (s[1] + 2*beta)
                est_vec[i] = torch.cat((est_vec[i], p[mask]))
        else: # Standard add-beta estimator
            # First half of sequence
            idx1 = idx[:jump-order+1]
            y1 = y[order-1:jump]
            for i in range(2**order):
                mask = (idx1 == i)
                s = y1[mask][:-1]
                p = (s.cumsum(0) + beta) / (torch.arange(1, len(s)+1, device=device) + 2*beta)
                p = F.pad(p, (1, 0), value=0.5)
                est_vec.append(p)
            # Second half of sequence
            idx2 = idx[jump-order+1:]
            y2 = y[jump:]
            for i in range(2**order):
                mask = (idx2 == i)
                s = y2[mask][:-1]
                p = (s.cumsum(0) + beta) / (torch.arange(1, len(s)+1, device=device) + 2*beta)
                p = F.pad(p, (1, 0), value=0.5)
                est_vec[i] = torch.cat((est_vec[i], p))
    else:
        if window in range(1, seq_length-order): # Windowed add-beta estimator
            for i in range(2**order):
                mask = (idx == i)
                s = torch.stack((y[order-1:] * mask.int(), mask.int()))
                s = F.pad(s, (window, 0))[:,:-1]
                s = F.conv1d(s, torch.ones(2, 1, window, device=device), groups=2)
                p = (s[0] + beta) / (s[1] + 2*beta)
                est_vec.append(p[mask])
        else: # Standard add-beta estimator
            counts = []
            totals = []
            for i in range(2**order):
                mask = (idx == i)
                s = y[order-1:][mask][:-1]
                count = s.cumsum(0)
                count = F.pad(count, (1, 0))
                total = torch.arange(len(s)+1, device=device)
                p = (count + beta) / (total + 2*beta)
                est_vec.append(p)
                counts.append(count)
                totals.append(total)
    if save_counts:
        return est_vec, counts, totals
    else:
        return est_vec

def optimal_est(P, type, order, sequence_length, generator, extra_args):
    if type == "jump-markov": # This is the optimal estimator knowing the jump point and the probability distributions
        jump = sequence_length // 2
        P1 = P[0]
        P2 = P[1]
        x, y = get_batch(P, type, order, sequence_length, 4096, generator, extra_args)
        powers = torch.Tensor([2**i for i in reversed(range(order))]).to(P.device)
        opt_logits = torch.zeros(x.size(0), x.size(1), P.size(1), device=P.device)
        if order > 1:
            opt_logits[:,:order-1,:] = 0.5*torch.ones(x.size(0), order-1, P.size(1), device=P.device)
        for i in range(order-1, jump-1):
            idx = x[:,i-order+1:i+1].float() @ powers
            opt_logits[:,i,:] = P1[idx.to(int)]
        for i in range(jump-1, sequence_length):
            idx = x[:,i-order+1:i+1].float() @ powers
            opt_logits[:,i,:] = P2[idx.to(int)]
        opt_logits = torch.log(opt_logits)
        opt_loss = F.nll_loss(opt_logits.view(-1, opt_logits.size(-1)), y.view(-1), ignore_index=-1)
    else:
        x, y = get_batch(P, type, order, sequence_length, 4096, generator, extra_args)
        powers = torch.Tensor([2**i for i in reversed(range(order))]).to(P.device)
        opt_logits = torch.zeros(x.size(0), x.size(1), P.size(1), device=P.device)
        if order > 1:
            opt_logits[:,:order-1,:] = 0.5*torch.ones(x.size(0), order-1, P.size(1), device=P.device)
        for i in range(order-1, sequence_length):
            idx = x[:,i-order+1:i+1].float() @ powers
            opt_logits[:,i,:] = P[idx.to(int)]
        opt_logits = torch.log(opt_logits)
        opt_loss = F.nll_loss(opt_logits.view(-1, opt_logits.size(-1)), y.view(-1), ignore_index=-1)

    return opt_loss

# Optimized Markov data generation (thank you @cekbote!)
def get_batch(P, type, order, seq_length, batch_size, generator, extra_args):
    data = torch.zeros(batch_size, seq_length+1, device=extra_args.device)
    if P == None:
        # Generate first k bits
        alpha = 0.5
        data[:, :order] = torch.bernoulli(alpha * torch.ones((batch_size, order), device=extra_args.device), generator=generator)
        # Generate following bits
        if type == "jump-markov":
            jump = seq_length // 2
            data[:, order:jump] = get_batch_from_past(data[:, :order], None, order, jump-order, batch_size, generator, extra_args.device, extra_args.dtype)
            data[:, jump:] = get_batch_from_past(data[:, :jump], None, order, seq_length-jump+1, batch_size, generator, extra_args.device, extra_args.dtype)
        else:
            data[:, order:] = get_batch_from_past(data[:, :order], None, order, seq_length-order+1, batch_size, generator, extra_args.device, extra_args.dtype)
    else:
        # Use same fixed P for all sequences
        if type == "jump-markov":
            # Generate first k bits
            alpha = 0.5
            data[:, :order] = torch.bernoulli(alpha * torch.ones((batch_size, order), device=extra_args.device), generator=generator)
            # Generate following bits
            jump = seq_length // 2
            data[:, order:jump] = get_batch_from_past(data[:, :order], P[0], order, jump-order, batch_size, generator, extra_args.device, extra_args.dtype)
            data[:, jump:] = get_batch_from_past(data[:, :jump], P[1], order, seq_length-jump+1, batch_size, generator, extra_args.device, extra_args.dtype)
        else:
            # Generate first k bits
            if extra_args.initial == 'steady':
                if P.size(0) == 2:
                    alpha = P[1,0] / (P[0,1] + P[1,0])
                else:
                    alpha = 0.5
            elif extra_args.initial == 'uniform':
                alpha = 0.5
            else:
                alpha = 0.5
            data[:, :order] = torch.bernoulli(alpha * torch.ones((batch_size, order), device=extra_args.device), generator=generator)
            # Generate following bits
            data[:, order:] = get_batch_from_past(data[:, :order], P, order, seq_length-order+1, batch_size, generator, extra_args.device, extra_args.dtype)
    x = data[:,:seq_length].to(int)
    y = data[:,1:].to(int)
    
    return x, y

def get_batch_from_past(past, P, order, seq_length, batch_size, generator, device, dtype):
    if P is None:
        P = get_random_P(order, batch_size, generator, device, dtype)
    else:
        P = P.unsqueeze(0).repeat(batch_size, 1, 1)
    data = torch.zeros(batch_size, order+seq_length, device=device)
    data[:,:order] = past[:,-order:]
    batch_indices = torch.arange(batch_size)
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(device)
    for i in range(order, seq_length):
        # Extract the previous 'order' symbols for the entire batch
        prev_symbols = data[:, i-order:i]
        # Compute indices using the dot product with powers of 2
        idx = (prev_symbols @ powers).int()
        # Fetch next symbols from the transition matrix P for each batch in parallel
        next_symbols = torch.multinomial(P[batch_indices, idx], 1, generator=generator).squeeze(1)
        data[:, i] = next_symbols

    return data[:,order:]


@torch.no_grad()
def eval(model, P, type, order, sequence_length, batch_size, generator, extra_args, max_num_batches=24, ctx=nullcontext()):
    assert model.training == False
    assert P is not None

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches):
        x, y = get_batch(P, type, order, sequence_length, batch_size, generator, extra_args)
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
def eval_probs(model, P, type, order, sequence_length, windows, generator, extra_args, betas = None, input_seq=None, output_seq=None, ctx=nullcontext()):
    assert model.training == False
    assert P is not None
    if betas is None:
        betas = [1]
    
    if input_seq is not None and output_seq is not None:
        x = input_seq[:, :sequence_length]
        y = output_seq[:, :sequence_length]
    else:
        x, y = get_batch(P, type, order, sequence_length, 1, generator, extra_args)

    # Get model estimation
    with ctx:
        outputs = model(x, targets=y, save_weights=True)
    probs = F.softmax(outputs['logits'], dim=-1)
    xb = x[0].float()
    probsb = probs[0, order-1:]
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(extra_args.device)
    idx = F.conv1d(xb.view(1, -1), powers.view(1, 1, -1)).squeeze()
    prob_vec = []
    for i in range(2**order):
        vec = probsb[idx == i][:,1] # estimated p
        prob_vec.append(vec)

    # Get (windowed) empirical add-beta estimator
    if windows is None:
        windows = [0]
    est_vec = []
    if windows == [0]:
        est_vec.append(empirical_est(x, y, type, order))
        beta_vec = []
        for beta in betas:
            beta_est = empirical_est(x, y, type, order, beta=beta)
            err = 0
            for i in range(2**order):
                err += torch.linalg.norm(prob_vec[i] - beta_est[i], ord=1)
            beta_vec.append(err)
    else:
        beta_vec = None
        for w in windows:
            est_vec.append(empirical_est(x, y, type, order, window=w))
    
    return prob_vec, est_vec, beta_vec

@torch.no_grad()
def eval_conditions(model, extra_args, ctx=nullcontext()):
    assert model.training == False

    x0 = torch.Tensor([[0,0,1,1,0]])
    x1 = torch.zeros(1,251)
    x = torch.cat((x0, x1), dim=1).to(int).to(extra_args.device)
    with ctx:
        outputs = model(x, targets=x, check_conditions=True)

    return None


def save_checkpoint(model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
