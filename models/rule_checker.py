import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_traj_direction(traj):
    traj_pad = nn.functional.pad(traj, [0, 0, 1, 1])
    traj_pad[..., 0, :] = 2 * traj_pad[..., 1, :] - traj_pad[..., 2, :]
    traj_pad[..., -1, :] = 2 * traj_pad[..., -2, :] - traj_pad[..., -3, :]

    appo_vel = (traj_pad[..., 2:, :] - traj_pad[..., :-2, :]) / 2
    appo_vel_norm = torch.norm(appo_vel, dim=-1, keepdim=True)
    traj_dir = appo_vel / (appo_vel_norm + 1e-06)

    # mask out the points where speed is too low
    traj_dir = traj_dir * (appo_vel_norm >= 2).float()

    return traj_dir

def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (torch.pi / 2) + torch.sign(-yaw) * torch.abs(yaw)

def traj2patch(initial_node_pos, traj, patch_size):
    # initial_node_pos_xy = initial_node_pos[0:2]
    # head_idx = self.hyperparams['map_encoder'][self.node_type.name]['heading_state_index']
    # if type(head_idx) is list:  # infer from velocity or heading vector
    #     initial_node_head = torch.atan2(inputs[:, -1, head_idx[1]],
    #                                     inputs[:, -1, head_idx[0]])
    # else:
    #     initial_node_head = inputs[:, -1, head_idx]
    # yaw = initial_node_pos[-1]
    if type(initial_node_pos) is torch.Tensor:
        yaw = torch.tensor(initial_node_pos[..., -1].clone().detach(), device= traj.device)
    else:
        yaw = torch.tensor(initial_node_pos[-1], device= traj.device)
    if yaw.shape == torch.Size([]):
        yaw = yaw.unsqueeze(0)
    angle_in_rad = -angle_of_rotation(yaw)
    rot_mat = \
        torch.stack([torch.stack([torch.cos(angle_in_rad), torch.sin(angle_in_rad)]),
                        torch.stack([-torch.sin(angle_in_rad), torch.cos(angle_in_rad)])]).permute(2, 0, 1)
    patch_size = patch_size

    traj_patch = traj_fit_patch(traj, rot_mat, patch_size)
    return traj_patch

def traj_fit_patch(traj, rot_mat, patch_size):
    # traj = (traj.squeeze(0) - torch.tensor(init_pos)) * 3  # todo hardcode for nuscene. change it
    traj = traj.squeeze(0)* 2
    if traj.type() != rot_mat.type():
        rot_mat = rot_mat.type_as(traj)
    if len(traj.shape) == 4: # for pred traj
        rot_mat = rot_mat.unsqueeze(1)
    traj = torch.matmul(rot_mat.unsqueeze(1), traj.unsqueeze(-1)).squeeze(-1)

    traj[..., 1] += patch_size[0]
    traj[..., 0] += patch_size[1]
    return traj


def safety_checker(traj, safe_map, use_continuous_feature=False):
    if len(traj.shape) == 4:
        traj = traj.permute(1,0,2,3) # make batch at second place
    else:
        traj = traj.unsqueeze(1)
    traj_shape = traj.shape
    n_batch = traj_shape[-3]
    # safe_map = safe_map.permute(0, 3, 1, 2) # make sure safe_map shape is torch.Size([32, 3, 224, 224])
    _, _, h, w = safe_map.shape 

    # lane_info_map = torch.norm(safe_map[:, :2], dim=-1)

    traj_direction = get_traj_direction(traj)

    traj_round = torch.round(traj).long() # torch.Size([20, 32, 12, 2])
    # the points outside the map
    out_idx = (traj_round[..., 0] >= h) + (traj_round[..., 1] >= w) + (traj_round < 0).sum(dim=-1).type(torch.bool)
    traj_round[out_idx] = 0 

    idx = torch.cat([torch.zeros_like(traj_round)[..., :1], traj_round], dim=-1) #torch.Size([20, 32, 12, 3])
    idx[..., 0] = torch.arange(0, n_batch, device=traj_round.device)[:, None] # batch index
    idx = idx.view(-1, 3) # torch.Size([7680, 3]), [ 31, 115, 109] # a list of batch, coordinates

    # check lane direction, safe_map torch.Size([32, 3, 224, 224])
                            # batch ， channel, x, y
    lane_direction = safe_map[idx[:, 0], :2, idx[:, 1], idx[:, 2]].view(*traj_shape[:-1], 2).flip(dims=[-1]) #flip for direction dot product
    check_lane = torch.sum(traj_direction * lane_direction, dim=-1) < 0 #tensor(248, device='cuda:0') lane_direction全是0？
    if use_continuous_feature:

        # time_now = time.time()
        check_lane_step_c, safety_drivable_step_c = continuous_checker(traj_direction, lane_direction, safe_map, traj_shape, idx, out_idx, traj_round)
        # print('continuous_checker time comsumption = ', time.time() - time_now) 

        safety_per_step = (check_lane_step_c + safety_drivable_step_c) > 0
        safety_each_traj = safety_per_step.sum(-1)

        return safety_each_traj.bool(), safety_per_step.bool(), check_lane_step_c, safety_drivable_step_c

    # check drivable region
    drivable_region = safe_map[idx[:, 0], 2, idx[:, 1], idx[:, 2]].view(*traj_shape[:-1])
    check_drivable = drivable_region < 1 # tensor(461, device='cuda:0')
    safety_per_step = (check_lane + check_drivable) * ~out_idx  # True means no safe #tensor(462, device='cuda:0')
    safety_each_traj = safety_per_step.sum(-1)
    check_lane_step = check_lane * ~out_idx
    safety_drivable_step = check_drivable * ~out_idx
    # safety_step = safety_step
    

    return safety_each_traj.bool(), safety_per_step.bool(), check_lane_step, safety_drivable_step


def continuous_checker(traj_direction, lane_direction, safe_map, traj_shape, idx, out_idx, traj_round):
    
    # check lane direction
    max_dir = 2
    lane_direction = safe_map[idx[:, 0], :2, idx[:, 1], idx[:, 2]].view(*traj_shape[:-1], 2).flip(dims=[-1])
    direction_product = torch.sum(traj_direction * lane_direction, dim=-1)
    check_lane_position = direction_product < 0 #tensor(248, device='cuda:0')
    check_lane_c = (direction_product/max_dir) * check_lane_position.float()

    check_lane_step_c = -check_lane_c * (~out_idx).float() # element >0 means direction opposite, the more opposite the bigger element will be

    # check drivable region
    # drivable_region = safe_map[idx[:, 0], 2, idx[:, 1], idx[:, 2]].view(*traj_shape[:-1]) # torch.Size([20, 32, 12])
    # check_drivable = drivable_region < 1 # torch.Size([20, 32, 12]) True means no safe
    # dummy = torch.zeros(check_drivable.shape, device=check_drivable.device) # 
    safe_map_drivable = safe_map[:,2,:,:] / 255
    # indices_drivable = check_drivable.nonzero() # torch.Size([273, 3]), 

    # batch_idices = indices_drivable[:, 1]
    # tarj_idices = indices_drivable[:, 0]
    # point_idices = indices_drivable[:, 2]
    # #                               traj_idx ,              # batch_idx              # point_idx
    # drivable_positions = traj_round[tarj_idices, batch_idices, point_idices]

    conv_safe_map_drivable = get_conv_safe_map(safe_map_drivable) # safe_map_drivable, 1 means drivable
    dummy = conv_safe_map_drivable[idx[:, 0], idx[:, 1], idx[:, 2]].view(*traj_shape[:-1]) # find traj points on conv safemap
    check_drivable = dummy >= 1 # find drivable points
    check_non_drivable = ~check_drivable
    dummy = dummy - check_non_drivable.float()
    x = dummy < 0
    dummy = -(dummy * x)
    
    # time_now = time.time()

    # for batch_idx, tarj_idx, point_idx, position in zip(batch_idices, tarj_idices, point_idices, drivable_positions): # [1, 11], trj_idx, batch_idx, point_idx, safe_map torch.Size([32, 3, 224, 224])

    #     # top, down, left, right, top_left, top_right, down_left, down_right = 0,0,0,0,0,0,0,0

    #     safe_map_drivable_batch = conv_safe_map_drivable[batch_idx]

    #     dummy[tarj_idx, batch_idx, point_idx] = safe_map_drivable_batch[position[0], position[1]]


    #     # p0 = position[0]
    #     # p1 = position[1]
         
    #     # # top 3
    #     # if p0 > 0:
    #     #     top = safe_map_drivable_batch[p0 -1 , p1 ]
    #     #     top_left = safe_map_drivable_batch[p0 -1 , p1 -1 ] if (p1 > 0) else 0
    #     #     top_right = safe_map_drivable_batch[p0 -1 , p1 +1 ] if (p1 < 223) else 0
    #     # else:
    #     #     top = 0
        
    #     # # down 3
    #     # if p0 < 223:
    #     #     down = safe_map_drivable_batch[p0 +1 , p1 ]
    #     #     down_left = safe_map_drivable_batch[p0 +1 , p1 -1 ] if (p1 > 0) else 0
    #     #     down_right = safe_map_drivable_batch[p0 +1 , p1 +1 ] if (p1 < 223) else 0
    #     # else:
    #     #     down = 0

    #     # # left and right 2
    #     # left = safe_map_drivable_batch[p0  , p1 -1 ] if p1 > 0 else 0
    #     # right = safe_map_drivable_batch[p0  , p1 +1 ] if p1 < 223 else 0

    #     # dummy[tarj_idx, batch_idx, point_idx] = sum([top, down, left, right, top_left, top_right, down_left, down_right]) / 8
    
    safety_drivable_step_c =  dummy * (~out_idx).float()
    # print('drivable_positions time comsumption = ', time.time() - time_now) 

    # safety_step = (check_lane_c + check_drivable) * ~out_idx  # True means no safe #tensor(462, device='cuda:0')
    # safety = safety_step.sum(-1)
    # safety_step = safety_step

    return check_lane_step_c, safety_drivable_step_c


def get_conv_safe_map(safe_map):
    
    kernel = [[1, 1, 1],
             [1, 0, 1],
             [1, 1, 1]]

    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=kernel, requires_grad=False).to(safe_map.device)

    conv_safe_map = torch.nn.functional.conv2d(safe_map.unsqueeze(1), weight, padding=1)

    filter_safe_map = safe_map + (conv_safe_map / 9).squeeze(1) # (1 + (a > 0)) > 0, safe point always > 1, no safe max value is 8/9
    
    return filter_safe_map

def min_ade(traj: torch.Tensor, traj_gt: torch.Tensor):
    """
    Computes average displacement error for the best trajectory is a set, with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape [batch_size]
    """
    traj = traj.permute(2, 0, 1, 3)
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    # masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt[:,:,:traj.shape[2],:] - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.pow(err, exponent=0.5)
    err = torch.sum(err , dim=2)
    err, inds = torch.min(err, dim=1)

    return err, inds


