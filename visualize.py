import random
import numpy as np
import torch
from datasets.argoverse.dataset import ArgoH5Dataset
from datasets.interaction_dataset.dataset import InteractionDataset
from datasets.nuscenes.dataset import NuscenesH5Dataset
from datasets.trajnetpp.dataset import TrajNetPPDataset
from models.autobot_ego import AutoBotEgo 
from models.autobot_ego_gan import AutoBotEgoGan 
from models.autobot_joint import AutoBotJoint
from nuscenes.prediction import PredictHelper
from process_args import get_vis_args
from utils.metric_helpers import min_xde_K, yaw_from_predictions, interpolate_trajectories, collisions_for_inter_dataset
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
import os
import imageio
import matplotlib.pyplot as plt
from typing import Dict, Union, List
from models.autobot_ego_gan import safety_checker, traj2patch
import sys
sys.path.append('../')
# from train_eval.initialization import initialize_prediction_model, initialize_dataset, get_specific_args
from nuscenes import NuScenes
import torch.nn as nn

class Visualizer:
    def __init__(self, args, model_config, model_dirname):
        self.args = args
        self.data_root = self.args.data_root
        self.model_config = model_config
        self.model_dirname = model_dirname
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        torch.manual_seed(self.model_config.seed)
        if torch.cuda.is_available() and not self.args.disable_cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.model_config.seed)
        else:
            self.device = torch.device("cpu")

        self.interact_eval = False  # for evaluating on the interaction dataset, we need this bool.
        self.initialize_dataloader()
        self.initialize_model()

        # Initialize epochs
        self.current_epoch = 0
        self.unlike_slow_start_epoch = 5
        self.unlike_slow_start_end_epoch = 10
        self.start_gamma = 0
        self.end_gamma = 1
        self.map_extent = [ -50, 50, -20, 80 ] 
        self.canvas_size = [224, 224]
        # test_set = initialize_dataset(ds_type, ['load_data', self.data_root, cfg['test_set_args']] + spec_args)
        # self.ds = test_set
        # nuscenes = NuScenesDatasetSafe(data_root=args.raw_dataset_path, split_name=args.split_name,
        #                        version='v1.0-trainval', ego_range=args.ego_range, num_others=max_num_agents)

    def get_gamma(self, idx_batch):
        """
        get current gamma for unlike loss weight
        """
        gamma = self.start_gamma
        if self.current_epoch >=self.unlike_slow_start_epoch and self.current_epoch <= self.unlike_slow_start_end_epoch:
            gamma = ((idx_batch + 1) / ((self.unlike_slow_start_end_epoch - self.unlike_slow_start_epoch + 1) * \
            self.dl_tr_length)) * (self.current_epoch - self.unlike_slow_start_epoch + 1) * self.end_gamma
        if self.current_epoch >self.unlike_slow_start_end_epoch:
            gamma = self.end_gamma
        return gamma

    def initialize_dataloader(self):
        if "Nuscenes" in self.model_config.dataset:
            val_dset = NuscenesH5Dataset(dset_path=self.args.dataset_path, split_name="val",
                                         model_type=self.model_config.model_type,
                                         use_map_img=self.model_config.use_map_image,
                                         use_map_lanes=self.model_config.use_map_lanes,
                                         rtn_extras=True)

        elif "interaction-dataset" in self.model_config.dataset:
            val_dset = InteractionDataset(dset_path=self.args.dataset_path, split_name="val",
                                          use_map_lanes=self.model_config.use_map_lanes, evaluation=True)
            self.interact_eval = True

        elif "trajnet++" in self.model_config.dataset:
            val_dset = TrajNetPPDataset(dset_path=self.model_config.dataset_path, split_name="val")

        elif "Argoverse" in self.model_config.dataset:
            val_dset = ArgoH5Dataset(dset_path=self.args.dataset_path, split_name="val",
                                     use_map_lanes=self.model_config.use_map_lanes)

        else:
            raise NotImplementedError

        self.num_other_agents = val_dset.num_others
        self.pred_horizon = val_dset.pred_horizon
        self.k_attr = val_dset.k_attr
        self.map_attr = val_dset.map_attr
        self.predict_yaw = val_dset.predict_yaw
        self.val_dset = val_dset
        if "Joint" in self.model_config.model_type:
            self.num_agent_types = val_dset.num_agent_types

        self.val_loader = torch.utils.data.DataLoader(
            val_dset, 1, shuffle=True, num_workers=12, drop_last=False,
            pin_memory=False
        )

        print("Val dataset loaded with length", len(val_dset))

    def initialize_model(self):
        if "Ego" in self.model_config.model_type:
            if "Gan" in self.model_config.model_type:
                self.autobot_model = AutoBotEgoGan(k_attr=self.k_attr,
                                d_k=self.model_config.hidden_size,
                                _M=self.num_other_agents,
                                c=self.model_config.num_modes,
                                T=self.pred_horizon,
                                L_enc=self.model_config.num_encoder_layers,
                                dropout=self.model_config.dropout,
                                num_heads=self.model_config.tx_num_heads,
                                L_dec=self.model_config.num_decoder_layers,
                                tx_hidden_size=self.model_config.tx_hidden_size,
                                use_map_img=self.model_config.use_map_image,
                                use_map_lanes=self.model_config.use_map_lanes,
                                map_attr=self.map_attr,
                                entropy_weight = self.model_config.entropy_weight,
                                kl_weight = self.model_config.kl_weight,
                                use_FDEADE_aux_loss = self.model_config.use_FDEADE_aux_loss).to(self.device)
            else:
                self.autobot_model = AutoBotEgo(k_attr=self.k_attr,
                                                d_k=self.model_config.hidden_size,
                                                _M=self.num_other_agents,
                                                c=self.model_config.num_modes,
                                                T=self.pred_horizon,
                                                L_enc=self.model_config.num_encoder_layers,
                                                dropout=self.model_config.dropout,
                                                num_heads=self.model_config.tx_num_heads,
                                                L_dec=self.model_config.num_decoder_layers,
                                                tx_hidden_size=self.model_config.tx_hidden_size,
                                                use_map_img=self.model_config.use_map_image,
                                                use_map_lanes=self.model_config.use_map_lanes,
                                                map_attr=self.map_attr).to(self.device)

        elif "Joint" in self.model_config.model_type:
            self.autobot_model = AutoBotJoint(k_attr=self.k_attr,
                                              d_k=self.model_config.hidden_size,
                                              _M=self.num_other_agents,
                                              c=self.model_config.num_modes,
                                              T=self.pred_horizon,
                                              L_enc=self.model_config.num_encoder_layers,
                                              dropout=self.model_config.dropout,
                                              num_heads=self.model_config.tx_num_heads,
                                              L_dec=self.model_config.num_decoder_layers,
                                              tx_hidden_size=self.model_config.tx_hidden_size,
                                              use_map_lanes=self.model_config.use_map_lanes,
                                              map_attr=self.map_attr,
                                              num_agent_types=self.num_agent_types,
                                              predict_yaw=self.predict_yaw).to(self.device)
        else:
            raise NotImplementedError

        model_dicts = torch.load(self.args.model_path, map_location=self.device)
        self.autobot_model.load_state_dict(model_dicts["AutoBot"])
        self.autobot_model.eval()

        model_parameters = filter(lambda p: p.requires_grad, self.autobot_model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of Model Parameters:", num_params)

    def _data_to_device(self, data):
        if "Joint" in self.model_config.model_type:
            ego_in, ego_out, agents_in, agents_out, context_img, agent_types = data
            ego_in = ego_in.float().to(self.device)
            ego_out = ego_out.float().to(self.device)
            agents_in = agents_in.float().to(self.device)
            agents_out = agents_out.float().to(self.device)
            context_img = context_img.float().to(self.device)
            agent_types = agent_types.float().to(self.device)
            return ego_in, ego_out, agents_in, agents_out, context_img, agent_types

        elif "Ego" in self.model_config.model_type:
            ego_in, ego_out, agents_in, roads, extras, init_pose, safe_map  = data # in_ego, out_ego, in_agents, ego_roads, extras, global_pose, safe_maps
            ego_in = torch.tensor(ego_in).float().to(self.device)
            ego_out = torch.tensor(ego_out).float().to(self.device)
            agents_in = torch.tensor(agents_in).float().to(self.device)
            roads = torch.tensor(roads).float().to(self.device)
            safe_map = torch.tensor(safe_map).float().to(self.device)
            init_pose = torch.tensor(init_pose).float().to(self.device)
            instance_token = extras[2]
            sample_token = extras[3]
            return ego_in, ego_out, agents_in, roads, safe_map, init_pose, instance_token, sample_token

    # Visualize
    def visualize(self, output_dir: str, dataset_type: str):
        """
        Generate visualizations for predictions
        :param output_dir: results directory to dump visualizations in
        :param dataset_type: e.g. 'nuScenes'. Visualizations will vary based on dataset.
        :return:
        """
        if dataset_type == 'nuScenes':
            self.visualize_nuscenes(output_dir)

    def visualize_nuscenes(self, output_dir):
        index_list = self.get_vis_idcs_nuscenes()
        resolution = 0.1
        ds = NuScenes('v1.0-trainval', dataroot=self.data_root)
        ds_helper = PredictHelper(ds)
        store_3_in_1 = False
        store_pred = True
        store_hd = False
        store_gt = False
        store_gif = False
        if not os.path.isdir(os.path.join(output_dir, 'vis_gif_im')):
            os.mkdir(os.path.join(output_dir, 'vis_gif_im'))
        if not os.path.isdir(os.path.join(output_dir, 'vis_gif_im', 'gifs')):
            os.mkdir(os.path.join(output_dir, 'vis_gif_im', 'gifs'))
        if not os.path.isdir(os.path.join(output_dir, 'vis_gif_im', 'imgs')):
            os.mkdir(os.path.join(output_dir, 'vis_gif_im', 'imgs'))
        for n, indices in enumerate(index_list):
            imgs, imgs_hd, imgs_pred, imgs_gt = self.generate_nuscenes_gif(indices, resolution, self.map_extent, ds_helper, ds, plot_pic_separetely=True)
            if store_gif:
                filename = os.path.join(output_dir, 'vis_gif_im', 'gifs', 'example' + str(n) + '.gif')
                imageio.mimsave(filename, imgs, format='GIF', fps=1)
            for j, img in enumerate(imgs):
                img_name = os.path.join(output_dir, 'vis_gif_im', 'imgs', 'example' + str(n) + '_' + str(j))
                if store_3_in_1:
                    imageio.imsave(img_name + '.png', img)
                if store_gt:
                    imageio.imsave(img_name + 'gt' + '.png', imgs_gt[j])
                if store_hd:
                    imageio.imsave(img_name + 'hd' + '.png', imgs_hd[j])                    
                if store_pred:
                    imageio.imsave(img_name + 'pred' + '.png', imgs_pred[j])
                # imageio.imwrite(img_name + "wirte" + '.png', img, quality=100)

    def get_vis_idcs_nuscenes(self):
        """
        Returns list of list of indices for generating gifs for the nuScenes val set.
        Instance tokens are hardcoded right now. I'll fix this later (TODO)
        """
        token_list = get_prediction_challenge_split('val', dataroot=self.data_root)
        instance_tokens = [token_list[idx].split("_")[0] for idx in range(len(token_list))]
        unique_instance_tokens = []
        for i_t in instance_tokens:
            if i_t not in unique_instance_tokens:
                unique_instance_tokens.append(i_t)

        # instance_tokens_to_visualize = [54, 98, 91, 5, 114, 144, 291, 204, 312, 187, 36, 267, 146]
        # instance_tokens_to_visualize = np.random.randint(500,size=100)
        instance_tokens_to_visualize = np.arange(4,5)

        idcs = []
        for i_t_id in instance_tokens_to_visualize:
            idcs_i_t = [i for i in range(len(instance_tokens)) if instance_tokens[i] == unique_instance_tokens[i_t_id]]
            idcs.append(idcs_i_t)
        print('starting visualizin {} examples '.format(len(idcs)))
        return idcs

    def generate_nuscenes_gif(self, idcs: List[int], resolution, map_extent, ds_helper, ds, plot_pic_separetely=False):
        """
        Generates gif of predictions for the given set of indices.
        :param idcs: val set indices corresponding to a particular instance token.
        """

        # Raster maps for visualization.
        static_layer_rasterizer = StaticLayerRasterizer(ds_helper,
                                                        resolution=resolution,
                                                        meters_ahead=map_extent[3],
                                                        meters_behind=-map_extent[2],
                                                        meters_left=-map_extent[0],
                                                        meters_right=map_extent[1])

        agent_rasterizer = AgentBoxesWithFadedHistory(ds_helper, seconds_of_history=1,
                                                      resolution=resolution,
                                                      meters_ahead=map_extent[3],
                                                      meters_behind=-map_extent[2],
                                                      meters_left=-map_extent[0],
                                                      meters_right=map_extent[1])

        raster_maps = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

        imgs = []
        imgs_hd = []
        imgs_pred = []
        imgs_gt = []

        # build a color map
        # top = cm.get_cmap('Oranges_r', 128)
        # bottom = cm.get_cmap('Blues', 128)

        # newcolors = np.vstack((top(np.linspace(0, 1, 128)),
        #                     bottom(np.linspace(0, 1, 128))))
        # newcmp = ListedColormap(newcolors, name='OrangeBlue')
        # traj_cmap = plt.get_cmap('OrRd')
        traj_cmap = plt.cm.OrRd
        traj_norm = plt.Normalize(vmin=0, vmax=0.5)

        for idx in idcs:

            # Load data
            data = self.val_dset[idx]
            ego_in, ego_out, agents_in, roads, safemap, init_pose, instance_token, sample_token  = self._data_to_device(data)

            # input_keys = data['inputs']['map_representation'].keys()
            # map_representation = data['inputs']['map_representation']
            # if 'safe_map' in input_keys:
            #     safemap = map_representation['safe_map']
            # if 'safety_map' in input_keys:
            #     safemap = map_representation['safety_map']
            # if 'curretnt_global_pose' in input_keys:
            #     ini_pose = map_representation['curretnt_global_pose']
            # if 'global_pose' in input_keys:
            #     ini_pose = map_representation['global_pose']

            # Get raster map
            hd_map = raster_maps.make_input_representation(instance_token, sample_token)

            # transfer black pixel to gray
            black_pixels = np.where(
                (hd_map[:, :, 0] == 0) & 
                (hd_map[:, :, 1] == 0) & 
                (hd_map[:, :, 2] == 0)
            )
            hd_map[black_pixels] = [205, 193, 197]

            r, g, b = hd_map[:, :, 0] / 255, hd_map[:, :, 1] / 255, hd_map[:, :, 2] / 255
            hd_map_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

            # Predict
            gamma = self.get_gamma(idx)

            ego_in, agents_in, roads, ego_out, safemap, init_pose = self.reshape_data_to_batch([ego_in, agents_in, roads, ego_out, safemap, init_pose])

            # permute safe map in ([bz, w, h, channel]) to bz, channel, w, h ([32, 3, 224, 224])
            # safemap = safemap.permute(0, 2, 3, 1)

            # encode observations
            if "Gan" in self.model_config.model_type:
                pred_obs, mode_probs, violation_rate = \
                    self.autobot_model(ego_in, agents_in, roads, ego_out, safemap, init_pose, gamma)
                # val_violation_rates.append(violation_rate.cpu().numpy())
            else:
                pred_obs, mode_probs = \
                    self.autobot_model(ego_in, agents_in, roads)


            # Plot
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(hd_map, extent=map_extent)
            ax[1].imshow(hd_map_gray, cmap='gist_gray', extent=map_extent)
            ax[2].imshow(hd_map_gray, cmap='gist_gray', extent=map_extent)

            if plot_pic_separetely:
                fig_hd = plt.figure(figsize=(5, 5), dpi=500)
                ax_hd = plt.gca() 
                ax_hd.imshow(hd_map, extent=map_extent)

                fig_gray_pred = plt.figure(figsize=(5, 5), dpi=500)
                ax_gray_pred = plt.gca()
                ax_gray_pred.imshow(hd_map_gray, cmap='gist_gray', extent=map_extent) 

                fig_gray_gt = plt.figure(figsize=(5, 5), dpi=500)
                ax_gray_gt = plt.gca() 
                ax_gray_gt.imshow(hd_map_gray, cmap='gist_gray', extent=map_extent) 
            # traj_pred = predictions['traj_k'] if 'traj_k' in predictions.keys() else predictions['traj']
            # if 'probs_k' in predictions:
            #     log_prob_pred = predictions['probs_k']
            # else:
            #     log_prob_pred = predictions['probs']
            # prob_pred = torch.exp(log_prob_pred)

            vio_flag = self.get_violation_index(pred_obs, ego_out, safemap, init_pose)
            
            # print(vio_flag)

            # visualization(pred_obs, safemap, 0, 0)

            for n, traj in enumerate(pred_obs):
                # ax[1].plot(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy(), lw=4,
                #            color='r', alpha=0.8)
                # ax[1].scatter(traj[-1, 0].detach().cpu().numpy(), traj[-1, 1].detach().cpu().numpy(), 60,
                #               color='r', alpha=0.8)
                # ax[1].plot(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy(), lw=4,
                #         color=prob_pred[:,n].item(), cmap=traj_cmap, norm=traj_norm, alpha=0.8)
                # ax[1].scatter(traj[-1, 0].detach().cpu().numpy(), traj[-1, 1].detach().cpu().numpy(), 60,
                #             color=prob_pred[:,n].item(), cmap=traj_cmap, norm=traj_norm, alpha=0.8)
                traj = traj.squeeze(1)
                if vio_flag[n] == False:
                    color = 'r'
                else:
                    color = 'blue'
                ax[1].plot(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy(), lw=2,
                        color=color, alpha=0.8)
                ax[1].scatter(traj[-1, 0].detach().cpu().numpy(), traj[-1, 1].detach().cpu().numpy(), 20,
                            color=color, alpha=0.8)
                if plot_pic_separetely:
                    ax_gray_pred.plot(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy(), lw=2,
                            color=color, alpha=0.8)
                    ax_gray_pred.scatter(traj[-1, 0].detach().cpu().numpy(), traj[-1, 1].detach().cpu().numpy(), 20,
                            color=color, alpha=0.8)
                # if n > 0:
                #     break

            traj_gt = ego_out[0, :, :2]
            gt_color = 'yellow'
            ax[2].plot(traj_gt[:, 0].detach().cpu().numpy(), traj_gt[:, 1].detach().cpu().numpy(), lw=2, color=gt_color)
            ax[2].scatter(traj_gt[-1, 0].detach().cpu().numpy(), traj_gt[-1, 1].detach().cpu().numpy(), 20, color=gt_color)

            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            fig.tight_layout(pad=0)
            ax[0].margins(0)
            ax[1].margins(0)
            ax[2].margins(0)

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            imgs.append(image_from_plot)
            plt.close(fig)

            if plot_pic_separetely:
                ax_gray_gt.plot(traj_gt[:, 0].detach().cpu().numpy(), traj_gt[:, 1].detach().cpu().numpy(), lw=2, color=gt_color)
                ax_gray_gt.scatter(traj_gt[-1, 0].detach().cpu().numpy(), traj_gt[-1, 1].detach().cpu().numpy(), 20, color=gt_color)

                ax_hd.axis('off')
                ax_gray_pred.axis('off')
                ax_gray_gt.axis('off')
                fig_hd.tight_layout(pad=0)
                fig_gray_pred.tight_layout(pad=0)
                fig_gray_gt.tight_layout(pad=0)
                ax_hd.margins(0)
                ax_gray_pred.margins(0)
                ax_gray_gt.margins(0)

                fig_hd.canvas.draw()
                fig_gray_pred.canvas.draw()
                fig_gray_gt.canvas.draw()

                image_from_plot_hd = np.frombuffer(fig_hd.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot_pred = np.frombuffer(fig_gray_pred.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot_gt = np.frombuffer(fig_gray_gt.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot_hd = image_from_plot_hd.reshape(fig_hd.canvas.get_width_height()[::-1] + (3,))
                image_from_plot_pred = image_from_plot_pred.reshape(fig_gray_pred.canvas.get_width_height()[::-1] + (3,))
                image_from_plot_gt = image_from_plot_gt.reshape(fig_gray_gt.canvas.get_width_height()[::-1] + (3,))

                imgs_hd.append(image_from_plot_hd)
                imgs_pred.append(image_from_plot_pred)
                imgs_gt.append(image_from_plot_gt)
                plt.close(fig_hd)
                plt.close(fig_gray_pred)
                plt.close(fig_gray_gt)

        return imgs, imgs_hd, imgs_pred, imgs_gt

    def reshape_data_to_batch(self, data):

        ego_in, agents_in, roads, ego_out, safemap, ini_pose = data

        ego_in = ego_in.unsqueeze(0)
        agents_in = agents_in.unsqueeze(0)
        roads = roads.unsqueeze(0)
        ego_out = ego_out.unsqueeze(0)
        safemap = safemap.unsqueeze(0)
        ini_pose = ini_pose.unsqueeze(0)

        return ego_in, agents_in, roads, ego_out, safemap, ini_pose

    def get_violation_index(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor], safemap: torch.Tensor, ini_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute get_violation_index
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :param safemap: bz, w, h, channel
        :return:
        """
        # Unpack arguments
        # mu_x = pred_dist[:, :, 0]
        # mu_y = pred_dist[:, :, 1]

        predictions = predictions.permute(2, 0, 1, 3)
        traj_pred = predictions[..., :2]
        ground_truth = ground_truth[..., :2] # last colum is mask

        # Useful params
        batch_size = traj_pred.shape[0]
        num_pred_modes = traj_pred.shape[1]
        sequence_length = traj_pred.shape[2]
        traj_gt = ground_truth[:, :sequence_length, :]
        init_pose = ini_pose
        safe_mask = safemap

        image_center = [self.canvas_size[0]//2, self.canvas_size[1]//2] # image center
        traj_pred_fit = traj2patch(init_pose, traj_pred, image_center) # torch.Size([32, 10, 12, 2])
        # visualization(traj, torch.tensor(safe_mask).unsqueeze(0), idx, sample_idx=0)    
        safety_pred, safety_pred_step, _, _ = safety_checker(traj_pred_fit, safe_mask)

        # traj_gt_fit = traj2patch(init_pose, traj_gt.unsqueeze(1), image_center)
        # # visualization(traj, torch.tensor(safe_mask).unsqueeze(0), idx, sample_idx=0)    
        # safety_gt, safety_gt_step, _, _ = safety_checker(traj_gt_fit, safe_mask)
        
        # print('safety_gt_step', safety_gt_step[0])
        # visualization(traj_gt_fit, safe_mask, 0, sample_idx=0)  
        # print('safety_pred_step', safety_pred_step[0])  
        # visualization(traj_pred_fit, safe_mask, 0, sample_idx=0) 

        return safety_pred


def visualization(traj_total, safe_map_total, batch_idx=0, sample_idx=0):
    # :param traj_total: expected dimension [bz, mode_num, ph, xy]
    # :param safe_map_total: expected dimension [bz, channel, canvas_size, canvas_size]
    # :param batch_idx: which batch you want to visualize 
    # :param sample_idx: which mode you want to visualize 
    # :return:
    if len(traj_total.shape) == 4:
        traj = traj_total[batch_idx, sample_idx].detach().cpu()
    else:
        traj = traj_total[batch_idx].detach().cpu()
    safe_map = safe_map_total[batch_idx].detach().cpu()
    visualization_lane_direction(safe_map)
    plot_traj(traj)
    plt.show()


def visualization_lane_direction(safe_mask):
    direction_mask = safe_mask[:2].permute(1, 2, 0)
    drivable_mask = safe_mask[2]
    norm_x = np.round(np.linalg.norm(direction_mask, axis=-1))
    mask = np.stack([np.zeros_like(norm_x), drivable_mask, norm_x], axis=-1)
    # plt.imshow(mask, origin='lower')
    # plt.imshow(drivable_mask, origin='lower')
    a, b = np.where(norm_x > 0)
    select_idx = np.random.randint(0, len(a), 20) if len(a) > 0 else []
    # for idx in select_idx:
    #     plt.arrow(b[idx], a[idx], direction_mask[a[idx], b[idx]][0] * 5, direction_mask[a[idx], b[idx]][1] * 5,
    #               color='r', head_width=2)


def plot_traj(traj):
    appo_vel = get_traj_direction(traj)
    print('traj.shape', traj.shape)
    print('traj[0]', traj[0])
    plt.plot(traj[:, 1], traj[:, 0], 'o-b', markersize=3)
    for i in range(len(traj)):
        if np.linalg.norm(appo_vel[i]) == 0: continue
        plt.arrow(traj[i, 1], traj[i, 0], appo_vel[i, 1] * 5, appo_vel[i, 0] * 5,
                  color='y', head_width=2)
        

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


if __name__ == '__main__':
    args, config, model_dirname = get_vis_args()
    vis = Visualizer(args, config, model_dirname)
    # vis.evaluate()
    vis.visualize(output_dir=model_dirname, dataset_type='nuScenes')