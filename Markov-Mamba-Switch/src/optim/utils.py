from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from contextlib import nullcontext


def get_random_P(order, batch_size, generator, device, dtype):
    pk = torch.rand((batch_size, 2**order, 1), generator=generator, dtype=dtype, device=device)
    pk = torch.cat([pk, 0.5*torch.ones((batch_size, 1, 1), dtype=dtype, device=device)], dim=1)
    P = torch.cat([1 - pk, pk], dim=2)

    return P

def empirical_est(x, y, order, beta=1):
    assert x.size(0) == 1
    assert beta > 0

    device = x.device
    x = x.float().squeeze()
    y = y.float().squeeze()
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(device)
    est_vec = defaultdict(list)
    switch = (x == 2).nonzero(as_tuple=True)[0]
    switch = F.pad(switch, (0, 1), value=x.size(0))

    start = 0
    for s in switch:
        if s - start >= order:
            xs = x[start:s]
            ys = y[start:s]
            idxs = F.conv1d(xs.view(1, -1), powers.view(1, 1, -1)).squeeze()
            for i in range(2**order):
                mask = (idxs == i)
                seq = ys[order-1:][mask][:-1]
                count = seq.cumsum(0)
                count = F.pad(count, (1, 0))
                total = torch.arange(len(count), device=device)
                p = (count + beta) / (total + 2*beta)
                est_vec[i].append(p)
        start = s + 1
    for i in range(2**order):
        est_vec[i] = torch.cat(est_vec[i])
    
    return est_vec

# Optimized Switch Markov data generation
def get_batch(order, seq_length, batch_size, generator, extra_args, get_loss=False):
    switch_p = extra_args.switch_probability # Switch probability
    clamp = 2 ** order
    data = torch.zeros(batch_size, seq_length+order+1, device=extra_args.device)
    if get_loss:
        loss = 0
    # Generate transition kernels
    P = get_random_P(order, batch_size, generator, extra_args.device, extra_args.dtype)
    # Generate switch positions
    switch = torch.bernoulli(switch_p * torch.ones((batch_size, seq_length+1), device=extra_args.device), generator=generator)
    # Initialize data
    data[:, :order] = clamp * torch.ones((batch_size, order), device=extra_args.device)
    # Generate following bits
    batch_indices = torch.arange(batch_size)
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(extra_args.device)
    for i in range(seq_length+1):
        # Extract the previous 'order' symbols for the entire batch
        prev_symbols = data[:, i:i+order]
        # Compute indices using the dot product with powers of 2
        idx = torch.clamp(prev_symbols @ powers, max=clamp).to(int)
        # Fetch next symbols from the transition matrix P for each batch in parallel
        next_symbols = torch.multinomial(P[batch_indices, idx], 1, generator=generator).squeeze(1)
        data[:, i+order] = next_symbols + clamp * switch[:, i]
        # Update loss
        if get_loss and i > 0:
            opt_logits = torch.log(P[batch_indices, idx])
            target = torch.clamp(data[:, i+order], max=extra_args.vocab_size-1).to(int)
            loss += F.nll_loss(opt_logits.view(-1, opt_logits.size(-1)), target.view(-1), ignore_index=-1, reduction='sum')
        # Update Markov kernels after switch
        n_switch = torch.sum(switch[:,i]).item()
        if n_switch > 0:
            P[switch[:,i].bool()] = get_random_P(order, int(n_switch), generator, extra_args.device, extra_args.dtype)
    # Clamp to alphabet size
    data = torch.clamp(data, max=extra_args.vocab_size-1)
    # Separate input from target
    x = data[:,order:seq_length+order].to(int)
    y = data[:,order+1:].to(int)
    # Return data (and loss)
    if get_loss:
        loss = loss / (batch_size * seq_length) # Normalize loss
        return x, y, loss
    else:
        return x, y


@torch.no_grad()
def eval(model, order, sequence_length, batch_size, generator, extra_args, max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches):
        x, y = get_batch(order, sequence_length, batch_size, generator, extra_args)
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
def eval_probs(model, order, sequence_length, generator, extra_args, betas = None, input_seq=None, output_seq=None, ctx=nullcontext()):
    assert model.training == False

    if betas is None:
        betas = [1]
    
    if input_seq is not None and output_seq is not None:
        x = input_seq[:, :sequence_length]
        y = output_seq[:, :sequence_length]
    else:
        x, y = get_batch(order, sequence_length, 1, generator, extra_args)

    # Get model estimation
    with ctx:
        outputs = model(x, targets=y, save_weights=True)
    prob = F.softmax(outputs['logits'], dim=-1)
    xb = x[0].float()
    probsb = prob[0,:,1] # estimated p
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(extra_args.device)
    prob_vec = defaultdict(list)
    switch = (xb == 2).nonzero(as_tuple=True)[0]
    switch = F.pad(switch, (0, 1), value=xb.size(0))

    start = 0
    for s in switch:
        if s - start >= order:
            xs = xb[start:s]
            probs = probsb[start:s]
            idxs = F.conv1d(xs.view(1, -1), powers.view(1, 1, -1)).squeeze()
            for i in range(2**order):
                mask = (idxs == i)
                vec = probs[order-1:][mask]
                prob_vec[i].append(vec)
        start = s + 1
    for i in range(2**order):
        prob_vec[i] = torch.cat(prob_vec[i])

    # Get empirical add-beta estimator
    est_vec = empirical_est(x, y, order)
    beta_vec = []
    for beta in betas:
        beta_est = empirical_est(x, y, order, beta=beta)
        err = 0
        total = 0
        for i in range(2**order):
            err += torch.linalg.norm(prob_vec[i] - beta_est[i], ord=1)
            total += len(prob_vec[i])
        beta_vec.append(err / total)
    
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
