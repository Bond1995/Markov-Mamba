from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from contextlib import nullcontext


def get_random_P(dist, vocab_size, order, batch_size, device, dtype):
    P = dist.sample((batch_size, vocab_size**order)).to(device, dtype)

    return P


def optimal_est(dist, P, vocab_size, order, sequence_length, generator, extra_args):
    x, y = get_batch(dist, P, vocab_size, order, sequence_length, 2048, generator, extra_args)
    powers = torch.Tensor([vocab_size**i for i in reversed(range(order))]).to(P.device)
    opt_logits = torch.zeros(x.size(0), x.size(1), vocab_size, device=P.device)
    if order > 1:
        alpha = 1 / vocab_size
        opt_logits[:,:order-1,:] = alpha*torch.ones(x.size(0), order-1, vocab_size, device=P.device)
    for i in range(order-1, sequence_length):
        idx = x[:,i-order+1:i+1].float() @ powers
        opt_logits[:,i,:] = P[idx.to(int)]
    opt_logits = torch.log(opt_logits)
    opt_loss = F.nll_loss(opt_logits.view(-1, opt_logits.size(-1)), y.view(-1), ignore_index=-1)

    return opt_loss

# Optimized Markov data generation
def get_batch(dist, P, vocab_size, order, seq_length, batch_size, generator, extra_args):
    data = torch.zeros(batch_size, seq_length+1, device=extra_args.device)
    alpha = 1.0 / vocab_size
    if P == None:
        # Generate first k bits
        data[:, :order] = torch.multinomial(alpha * torch.ones((batch_size, vocab_size), device=extra_args.device), order, replacement=True, generator=generator)
        # Generate following bits
        data[:, order:] = get_batch_from_past(data[:, :order], dist, None, vocab_size, order, seq_length-order+1, batch_size, generator, extra_args.device, extra_args.dtype)
    else:
        # Use same fixed P for all sequences
        # Generate first k bits
        data[:, :order] = torch.multinomial(alpha * torch.ones((batch_size, vocab_size), device=extra_args.device), order, replacement=True, generator=generator)
        # Generate following bits
        data[:, order:] = get_batch_from_past(data[:, :order], dist, P, vocab_size, order, seq_length-order+1, batch_size, generator, extra_args.device, extra_args.dtype)
    x = data[:,:seq_length].to(int)
    y = data[:,1:].to(int)
    
    return x, y

def get_batch_from_past(past, dist, P, vocab_size, order, seq_length, batch_size, generator, device, dtype):
    if P is None:
        P = get_random_P(dist, vocab_size, order, batch_size, device, dtype)
    else:
        P = P.unsqueeze(0).repeat(batch_size, 1, 1)
    data = torch.zeros(batch_size, order+seq_length, device=device)
    data[:,:order] = past[:,-order:]
    batch_indices = torch.arange(batch_size)
    powers = torch.Tensor([vocab_size**i for i in reversed(range(order))]).to(device)
    for i in range(order, seq_length):
        # Extract the previous 'order' symbols for the entire batch
        prev_symbols = data[:, i-order:i]
        # Compute indices using the dot product with powers of vocab_size
        idx = (prev_symbols @ powers).int()
        # Fetch next symbols from the transition matrix P for each batch in parallel
        next_symbols = torch.multinomial(P[batch_indices, idx], 1, generator=generator).squeeze(1)
        data[:, i] = next_symbols

    return data[:,order:]


@torch.no_grad()
def eval(model, dist, P, vocab_size, order, sequence_length, batch_size, generator, extra_args, max_num_batches=24, ctx=nullcontext()):
    assert model.training == False
    assert P is not None

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches):
        x, y = get_batch(dist, P, vocab_size, order, sequence_length, batch_size, generator, extra_args)
        with ctx:
            outputs = model(x, targets=y)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


def save_checkpoint(model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
