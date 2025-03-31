from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import time

from .utils import eval, get_batch, get_random_P, optimal_est, save_checkpoint


def train_base(model, opt, dist, P, vocab_size, order, scheduler, iterations, acc_steps, batch_size, sequence_length, generator, eval_freq, ckpt_path, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.float16)
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None
    
    print(f"Compiling model ...")
    model = torch.compile(model) # requires pytorch 2.0+

    if P is not None:
        P_test = P
        print("Markov transition matrix:")
        print(P)
    else:
        P_test = get_random_P(dist, vocab_size, order, 1, extra_args.device, extra_args.dtype).squeeze(0)
        print("Test Markov transition matrix:")
        print(P_test)
    
    # Optimal test loss
    opt_loss = optimal_est(dist, P_test, vocab_size, order, sequence_length, generator, extra_args)
    if extra_args.wandb:
        wandb.log({
            "val/opt_loss": opt_loss,
        })

    model.train()
    t0 = time.time()
    while itr < iterations:
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y = get_batch(dist, P, vocab_size, order, sequence_length, batch_size, generator, extra_args)
            with type_ctx:
                outputs = model(x, targets=y)
            loss = outputs['loss'] / acc_steps
            loss.backward()
            substep += 1

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            t1 = time.time()
            dt = t1 - t0

            model.eval()
            train_loss = loss.detach().cpu().item()
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
            val_acc, val_loss, val_perplexity = eval(model, dist, P_test, vocab_size, order, sequence_length, batch_size, generator, 
                                                     extra_args, max_num_batches=10, ctx=type_ctx)

            print_string = f"{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
            print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
            if scheduler is not None:
                print_string += f" [lr] {current_lr:.5f}"
            print(print_string)

            if extra_args.wandb:
                wandb.log({
                    "iter": itr,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/loss_gap": val_loss - opt_loss,
                    "val/perplexity": val_perplexity,
                    "val/acc": val_acc,
                    "lr": current_lr,
                })

            model.train()
            t0 = time.time()

    print(f"saving checkpoint to {ckpt_path}")
    save_checkpoint(model=model,
                    opt=opt,
                    scheduler=scheduler,
                    itr=itr,
                    ckpt_path=ckpt_path)

