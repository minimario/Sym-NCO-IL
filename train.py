import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
import pickle
import numpy as np
import wandb
#wandb.init(project="off-co-tsp", entity="alstn12088")


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device), num_equivariant_samples=0)
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=100, filename=opts.train_dataset, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
    


    with open(opts.labelpath, 'rb') as f:  
        data = pickle.load(f)

    # Data Pre-processing
    sol_list = []

    for i in range(100):
        sol_list.append(data[0][i][1])

    # for CVRP, may need to pad with zeros
    max_len = max([len(i) for i in sol_list])
    sol_list = [i + [0] * (max_len - len(i)) for i in sol_list]
    solution = np.array(sol_list)
    solution = torch.tensor(solution).long().cuda()

    # Dataloader for unlabeled data
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=1000, distribution=opts.data_distribution))
    training_dataloader_ul = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)


    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts, 
            solution[(batch_id)*opts.batch_size:(batch_id+1)*opts.batch_size],
            next(iter(training_dataloader_ul))
        )

        step += 1

    epoch_duration = time.time() - start_time

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        # logger.info('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

        avg_reward = validate(model, val_dataset, opts)
        wandb.log({"val avg cost": avg_reward})


    lr_scheduler.step()



###################################################################### 
# Problem Symmetric Transformation
###################################################################### 

def SR_transform(x, y, idx, dim=2):
    if idx < 0.5:
        phi = idx * 4 * math.pi
    else:
        phi = (idx - 0.5) * 4 * math.pi

    x = x - 1 / 2
    y = y - 1 / 2

    x_prime = torch.cos(phi).cuda() * x - torch.sin(phi).cuda() * y
    y_prime = torch.sin(phi).cuda() * x + torch.cos(phi).cuda() * y

    if idx < 0.5:
        dat = torch.cat((x_prime + 1 / 2, y_prime + 1 / 2), dim=dim)
    else:
        dat = torch.cat((y_prime + 1 / 2, x_prime + 1 / 2), dim=dim)
    return dat


def augment_xy_data_by_N_fold(problems, N, depot=None):
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    if depot is not None:
        x_depot = depot[:, :, [0]]
        y_depot = depot[:, :, [1]]
    idx = torch.rand(N - 1)

    for i in range(N - 1):

        problems = torch.cat((problems, SR_transform(x, y, idx[i])), dim=0)
        if depot is not None:
            depot = torch.cat((depot, SR_transform(x_depot, y_depot, idx[i])), dim=0)

    if depot is not None:
        return problems, depot.view(-1, 2)

    return problems

def augment_xy_data_by_N_fold_cvrp(problems, N):
    x = problems["loc"][:, :, [0]]
    y = problems["loc"][:, :, [1]]

    x_depot = problems["depot"][:, [0]]
    y_depot = problems["depot"][:, [1]]
    
    demand = problems["demand"]

    idx = torch.rand(N - 1)
    
    for i in range(N - 1):
        problems["loc"] = torch.cat((problems["loc"], SR_transform(x, y, idx[i])), dim=0)
        problems["depot"] = torch.cat((problems["depot"], SR_transform(x_depot, y_depot, idx[i], dim=1)), dim=0)
        problems["demand"] = torch.cat((problems["demand"], demand), dim=0)
    
    return problems
        


def augment(input, N):
    return augment_xy_data_by_N_fold(input, N)

def augment_cvrp(input, N):
    return augment_xy_data_by_N_fold_cvrp(input, N)

# Size transformation
def rand_sizing(x):
    ratio = (torch.rand(1).cuda() * 0.7) + 0.3
    return ratio * x

# Traslation transformation
def rand_traslation(x):
    x_trans = torch.rand(1).cuda() -0.5
    y_trans = torch.rand(1).cuda() -0.5

    x[:,:,0] += x_trans
    x[:,:,1] += y_trans
    return x

# Rotational & Reflectional transformation
def rand_rotation(problems):
    idx = torch.rand(1)    

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    problems = SR_transform(x, y, idx)
    return problems

# CVRP
def rand_sizing_cvrp(x):
    loc, depot = x
    ratio = (torch.rand(1).cuda() * 0.7) + 0.3
    return ratio * loc, ratio * depot

def rand_traslation_cvrp(x):
    loc, depot = x
    x_trans = torch.rand(1).cuda() -0.5
    y_trans = torch.rand(1).cuda() -0.5

    loc[:,:,0] += x_trans
    loc[:,:,1] += y_trans
    depot[:,0] += x_trans
    depot[:,1] += y_trans
    return loc, depot

# Rotational & Reflectional transformation
def rand_rotation_cvrp(x):
    loc, depot = x
    idx = torch.rand(1)    

    x = loc[:, :, [0]]
    y = loc[:, :, [1]]

    depot_x = depot[:, [0]]
    depot_y = depot[:, [1]]

    loc = SR_transform(x, y, idx)
    depot = SR_transform(depot_x, depot_y, idx, dim=1)
    return loc, depot

###################################################################### 
# Problem Symmetric Transformation
###################################################################### 



def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts, 
        solution, 
        unsup_batch
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)

    # supervised augmetation with symmetric transformation
    if opts.label_aug:
        if opts.problem == "cvrp":
            x = augment_cvrp(x, opts.num_input_augmentations)
        elif opts.problem == "tsp":
            x = augment(x, opts.num_input_augmentations)
        solution = solution.repeat(opts.num_input_augmentations, 1)

    # Supervised Imitation Learning
    if opts.num_equivariant_samples > 0:
        _, log_likelihood, ll_t_list = \
            model(x, 
                  action=solution, 
                  num_equivariant_samples=opts.num_equivariant_samples)
    else:
        _, log_likelihood = model(x, action=solution, num_equivariant_samples=0)
        
    if opts.num_equivariant_samples > 0:
        equivariant_loss = 0
        for ll_t in ll_t_list:
            equivariant_loss += -ll_t.mean()
        equivariant_loss /= opts.num_equivariant_samples

    # Unsupervised Psuedo-label based Imitation Learning
    if opts.consistancy_learning:
        x_unsup, _ = baseline.unwrap_batch(unsup_batch)
        x_unsup = move_to(x_unsup, opts.device)
        # Collecting unlabled action
        with torch.no_grad():
        
            _, _, pi = model(x_unsup, num_equivariant_samples=0, return_pi = True)

    
        
        batch_size = len(x_unsup)

        # Transformation of unlabled action
        index = int(batch_size/10)
        for i in range(10):
            if opts.problem == "cvrp":
                loc, depot = x_unsup["loc"][i*index:(i+1)*index], x_unsup["depot"][i*index:(i+1)*index]
                loc, depot = rand_traslation_cvrp(rand_sizing_cvrp(rand_rotation_cvrp((loc, depot)))) 
                x_unsup["loc"][i*index:(i+1)*index], x_unsup["depot"][i*index:(i+1)*index] = loc, depot
            elif opts.problem == "tsp":
                x_unsup[i*index:(i+1)*index] = rand_traslation(rand_sizing(rand_rotation(x_unsup[i*index:(i+1)*index])))


        # Psuedo-label based consistancy learning with Symmetric transformation
        _, log_likelihood_unsup = model(x_unsup, num_equivariant_samples=0, action=pi)
        unsup_loss = (-log_likelihood_unsup).mean()
        reinforce_loss = (-log_likelihood).mean()
        if opts.num_equivariant_samples > 0:
            loss = reinforce_loss + 0.1 * unsup_loss + opts.supervise_lambda * equivariant_loss
        else:
            loss = reinforce_loss + 0.1 * unsup_loss

    else:
        reinforce_loss = (-log_likelihood).mean()
        if opts.num_equivariant_samples > 0:
            loss = reinforce_loss + opts.supervise_lambda * equivariant_loss
        else:
            loss = reinforce_loss
        
    # log reinforce_loss to wandb
    wandb.log({"reinforce_loss": reinforce_loss.item()})
    if opts.num_equivariant_samples > 0:
        wandb.log({"equivariant_loss": equivariant_loss.item()})
    if opts.consistancy_learning:
        wandb.log({"unsup_loss": unsup_loss.item()})
        wandb.log({"nll_unsup": (-log_likelihood_unsup).mean().item()})
    wandb.log({"loss": loss.item()})

    #log nll
    wandb.log({"nll": (-log_likelihood).mean().item()})


    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

