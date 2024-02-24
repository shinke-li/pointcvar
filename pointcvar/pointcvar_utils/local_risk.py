import torch
from torch.distributions import Normal

def compute_average_distance(input_tensor, k):
    B, N, _ = input_tensor.shape

    # compute the distance matrix using the formula (x-y)^2 = x^2 + y^2 - 2xy
    square = torch.sum(input_tensor ** 2, dim=-1, keepdims=True) # B x N x 1
    inner_product = torch.matmul(input_tensor, input_tensor.transpose(-2, -1)) # B x N x N
    distance_matrix = torch.sqrt(square + square.transpose(-2, -1) - 2 * inner_product + 1e-6) # B x N x N

    # find the k nearest points and compute average distance
    _, indices = torch.topk(distance_matrix, k, largest=False, sorted=True, dim=-1) # B x N x k
    k_nearest_distances = torch.gather(distance_matrix, -1, indices) # B x N x k
    average_distances = torch.mean(k_nearest_distances, dim=-1) # B x N
    return average_distances

def local_geometric_risk(avg_dist,mode='gaussian'):
    if mode == 'gaussian':
        mean = torch.mean(avg_dist, dim=1, keepdims=True) # B x 1
        std = torch.std(avg_dist, dim=1, keepdims=True) # B x 1

        # Compute gaussian density
        gaussian_density = (1 / (std * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-0.5 * ((avg_dist - mean) / std) ** 2) # B x N
        return gaussian_density 
    
    elif mode == 'norm':
        l2_norm = torch.norm(avg_dist, dim=1, keepdim=True) # B x 1

            # Normalize average distances by the L2 norm
        return avg_dist / (l2_norm + 1e-7) 
    

def calculate_local_risk(points, k = 20, mode = 'gaussian'):
    dist = compute_average_distance(points,k)
    return local_geometric_risk(dist, mode)

def cal_geo_risk(data, k=20):
   
    data.requires_grad = True
    try:
        data.grad.zero_()
    except Exception as e:
        print(e)
    dist = compute_average_distance(data,k)
    #dist.mean().backward()
    std_values = torch.std(dist, dim=1)
    std_values.sum().backward()
    risk = torch.norm(data.grad, dim=-1)
    
    risk = risk/torch.unsqueeze(torch.norm(risk, dim=1), dim=-1)
    return risk