from typing import Dict, Optional, Tuple

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class ConcatEncodersMix(nn.Module):
    def __init__(
        self,
        rgb_static: DictConfig,
        proprio: DictConfig,
        device: torch.device,
        depth_static: Optional[DictConfig] = None,
        rgb_gripper: Optional[DictConfig] = None,
        depth_gripper: Optional[DictConfig] = None,
        tactile: Optional[DictConfig] = None,
        state_decoder: Optional[DictConfig] = None,
    ):
        super().__init__()
        self._latent_size = rgb_static.visual_features
        self.seg_num = rgb_static.num_seg # 15
        self.seg_size = rgb_static.seg_size # 2
        if rgb_gripper:
            self._latent_size += rgb_gripper.visual_features
        if depth_static:
            self._latent_size += depth_static.visual_features
        if depth_gripper:
            self._latent_size += depth_gripper.visual_features
        if tactile:
            self._latent_size += tactile.visual_features
        visual_features = self._latent_size
        # super ugly, fix this clip ddp thing in a better way
        if "clip" in rgb_static["_target_"]:
            self.rgb_static_encoder = hydra.utils.instantiate(rgb_static, device=device)
        else:
            self.rgb_static_encoder = hydra.utils.instantiate(rgb_static)
        self.depth_static_encoder = hydra.utils.instantiate(depth_static) if depth_static else None
        self.rgb_gripper_encoder = hydra.utils.instantiate(rgb_gripper) if rgb_gripper else None
        self.depth_gripper_encoder = hydra.utils.instantiate(depth_gripper) if depth_gripper else None
        self.tactile_encoder = hydra.utils.instantiate(tactile)
        self.proprio_encoder = hydra.utils.instantiate(proprio)
        if self.proprio_encoder:
            self._latent_size += self.proprio_encoder.out_features

        self.state_decoder = None
        if state_decoder:
            state_decoder.visual_features = visual_features
            state_decoder.n_state_obs = self.proprio_encoder.out_features
            self.state_decoder = hydra.utils.instantiate(state_decoder)

        self.current_visual_embedding = None
        self.current_state_obs = None

    @property
    def latent_size(self):
        return self._latent_size

    # uniformly segment the image into num_segments segments according to the second dim, and then randomly an element from each segment
    # input: imgs.shape = [b, s, c, h, w]
    # output: seg_imgs.shape = [b, num_segments, c, h, w]
    def segment_vis_img(self, imgs: torch.Tensor, num_segments: int) -> torch.Tensor:
        ## vision 1
        # seg_imgs = torch.chunk(imgs, num_segments, dim=1)
        # seg_imgs = [seg[:, torch.randint(seg.shape[1], (1,))] for seg in seg_imgs]
        # seg_imgs = torch.cat(seg_imgs, dim=1)
        
        ## version 2，the case where s cannot divide num_segments is considered
        b, s, c, h, w = imgs.shape
        segment_size = s // num_segments # each segment has s/num_segments 32/8 frames 
        # Initialize an empty tensor for the output
        seg_imgs = torch.empty((b, num_segments, c, h, w), device=imgs.device)
        
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size if i < num_segments - 1 else s
            segment = imgs[:, start:end, :, :, :]
            current_segment_size = end - start
            # Randomly select an index from each segment's sequence dimension
            random_indices = torch.randint(low=0, high=current_segment_size, size=(b, 1), device=imgs.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            random_indices = random_indices.expand(-1, -1, c, h, w)

            # Use torch.gather to select the random elements from each segment
            selected_imgs = torch.gather(segment, 1, random_indices)

            seg_imgs[:, i:i+1, :, :, :] = selected_imgs
        
        return seg_imgs
    
    
    # uniformly segment the image into num_segments segments according to the second dim
    # keep the first and last frames
    def segment_vis_img_except_for_FL_frame(self, imgs: torch.Tensor, num_segments: int) -> torch.Tensor:
        
        ## the case where s cannot divide num_segments is considered
        b, s, c, h, w = imgs.shape # s = 32
        total_frames = s - 2 # keep the first and last frames
        segment_size = total_frames // num_segments # each segment has total_frames/num_segments (30/15) frames 
        # Initialize an empty tensor for the output
        seg_imgs = torch.empty((b, num_segments+2, c, h, w), device=imgs.device)
        
        for i in range(num_segments):
            start = i * segment_size + 1 # skip the first frame
            end = start + segment_size if i < num_segments - 1 else s - 1 # skip the last frame
            segment = imgs[:, start:end, :, :, :]
            current_segment_size = segment.shape[1]
            
            # # Randomly select an index from each segment's sequence dimension
            random_indices = torch.randint(low=0, high=current_segment_size, size=(b,), device=imgs.device)
            selected_imgs = segment[torch.arange(b, device=imgs.device), random_indices,:,:,:]
            selected_imgs = selected_imgs.unsqueeze(1)
            seg_imgs[:, i+1:i+2, :, :, :] = selected_imgs
        
        seg_imgs[:, 0:1, :, :, :] = imgs[:, 0:1, :, :, :]  # Add the first frame to seg_imgs
        seg_imgs[:, -1:, :, :, :] = imgs[:, -1:, :, :, :]  # Add the last frame to seg_imgs
        
        return seg_imgs
    
    
    def segment_vis_img_with_specified_segment_size(self, imgs: torch.Tensor, segment_size: int) -> torch.Tensor:
        b, s, c, h, w = imgs.shape  # Extract dimensions
        
        total_frames = s - 2  # Excluding the first and last frames for segmentation
        num_segments = total_frames // segment_size  # Calculate the number of segments
        
        if total_frames < segment_size:
            return imgs
        
        # Check if the total frames can be perfectly divided by the segment size
        # If not, add an extra segment for the remaining frames
        if total_frames % segment_size != 0:
            num_segments += 1
        
        # Initialize an empty tensor for the output
        # +2 to include the first and last frames
        seg_imgs = torch.empty((b, num_segments + 2, c, h, w), device=imgs.device)
        
        for i in range(num_segments):
            start = i * segment_size + 1  # Skip the first frame
            end = start + segment_size
            # Ensure we don't go beyond the second to last frame
            end = min(end, s - 1)
            
            # Handling the case when there's not enough frames left for a full segment
            # Ensure at least one frame is selected
            current_segment_size = max(1, end - start)
            segment = imgs[:, start:end, :, :, :]
            
            # Generate random indices for selecting frames within the current segment
            random_indices = torch.randint(low=0, high=current_segment_size, size=(b,), device=imgs.device)
            selected_imgs = segment[torch.arange(b, device=imgs.device), random_indices, :, :, :]
            selected_imgs = selected_imgs.unsqueeze(1)  # Add back the temporal dimension for concatenation
            seg_imgs[:, i+1:i+2, :, :, :] = selected_imgs
        
        # Add the first and last frames back to the segmented images
        seg_imgs[:, 0:1, :, :, :] = imgs[:, 0:1, :, :, :]
        seg_imgs[:, -1:, :, :, :] = imgs[:, -1:, :, :, :]
        
        return seg_imgs

    def segment_vis_img_include_FL_frame_stride_2_vectorize(self, imgs: torch.Tensor, segment_size: int) -> torch.Tensor:
        b, s, c, h, w = imgs.shape
        num_segments = (s + segment_size - 1) // segment_size  # Using ceiling division to handle last segment

        # Pre-allocating the output tensor
        seg_imgs = torch.empty((b, num_segments, c, h, w), device=imgs.device)
        
        # Vectorized calculation of segment starts and ends
        segment_starts = torch.arange(0, num_segments * segment_size, segment_size, device=imgs.device)
        segment_ends = segment_starts + segment_size
        segment_ends = torch.clamp(segment_ends, max=s)
        
        # Calculate current segment sizes and handle segments with not enough frames
        segment_sizes = segment_ends - segment_starts
        segment_sizes = torch.clamp(segment_sizes, min=1)  # Ensure at least one frame per segment

        # Generate random indices for each segment in a vectorized manner
        random_indices = torch.randint(low=0, high=segment_size, size=(b, num_segments), device=imgs.device)
        random_indices += segment_starts[None, :]  # Adjust indices to the global position
        random_indices = torch.clamp(random_indices, max=s-1)  # Ensure indices are within bounds

        # Use advanced indexing to select images directly into the output tensor
        arange_b = torch.arange(b, device=imgs.device).reshape(b, 1)
        selected_imgs = imgs[arange_b, random_indices, :, :, :].unsqueeze(2)  # Add dimension for concat
        
        # Fill the pre-allocated output tensor with selected images
        seg_imgs[:, :num_segments, :, :, :] = selected_imgs.squeeze(2)  # Remove the extra dimension added

        return seg_imgs


    def forward(
        self, imgs: Dict[str, torch.Tensor], depth_imgs: Dict[str, torch.Tensor], state_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_static = imgs["rgb_static"]
        rgb_gripper = imgs["rgb_gripper"] if "rgb_gripper" in imgs else None
        rgb_tactile = imgs["rgb_tactile"] if "rgb_tactile" in imgs else None
        depth_static = depth_imgs["depth_static"] if "depth_static" in depth_imgs else None
        depth_gripper = depth_imgs["depth_gripper"] if "depth_gripper" in depth_imgs else None

        # divide the original video imgs into self.seg_num segments, and then encode each segment. Stack the encoded segment features
        # rgb_static_seg = self.segment_vis_img(rgb_static, self.seg_num)
        # rgb_static_seg = self.segment_vis_img_except_for_FL_frame(rgb_static, self.seg_num) # keep the first and last frames
        rgb_static_seg = self.segment_vis_img_include_FL_frame_stride_2_vectorize(rgb_static, self.seg_size) # keep the first and last frames
        b, s_s, c, h, w = rgb_static_seg.shape # s_s = self.seg_num+2
        rgb_static_seg = rgb_static_seg.reshape(-1, c, h, w)
        encoded_imgs_seg = self.rgb_static_encoder(rgb_static_seg) # (batch*s_s, 64)， vision_network encoder
        encoded_imgs_seg = encoded_imgs_seg.reshape(b, s_s, -1)
        
        # original video imgs
        b, s, c, h, w = rgb_static.shape
        rgb_static = rgb_static.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 200, 200)
        # ------------ Vision Network ------------ #
        encoded_imgs = self.rgb_static_encoder(rgb_static)  # (batch*seq_len, 64)， vision_network encoder
        encoded_imgs = encoded_imgs.reshape(b, s, -1)  # (batch, seq, 64)

        if depth_static is not None:
            depth_static = torch.unsqueeze(depth_static, 2)
            depth_static = depth_static.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 3, 200, 200)
            encoded_depth_static = self.depth_static_encoder(depth_static)  # (batch*seq_len, 64)
            encoded_depth_static = encoded_depth_static.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_depth_static], dim=-1)

        if rgb_gripper is not None:
            
            # segment the gripper image
            # rgb_gripper_seg = self.segment_vis_img(rgb_gripper, self.seg_num)
            # rgb_gripper_seg = self.segment_vis_img_except_for_FL_frame(rgb_gripper, self.seg_num)
            rgb_gripper_seg = self.segment_vis_img_include_FL_frame_stride_2_vectorize(rgb_gripper, self.seg_size)
            b, s_s, c, h, w = rgb_gripper_seg.shape # s_s = self.seg_num
            rgb_gripper_seg = rgb_gripper_seg.reshape(-1, c, h, w)
            encoded_imgs_gripper_seg = self.rgb_gripper_encoder(rgb_gripper_seg)
            encoded_imgs_gripper_seg = encoded_imgs_gripper_seg.reshape(b, s_s, -1)
            encoded_imgs_seg = torch.cat([encoded_imgs_seg, encoded_imgs_gripper_seg], dim=-1)
            
            # original gripper video imgs
            b, s, c, h, w = rgb_gripper.shape
            rgb_gripper = rgb_gripper.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_imgs_gripper = self.rgb_gripper_encoder(rgb_gripper)  # (batch*seq_len, 64)
            encoded_imgs_gripper = encoded_imgs_gripper.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_imgs_gripper], dim=-1)
            
            
            if depth_gripper is not None:
                depth_gripper = torch.unsqueeze(depth_gripper, 2)
                depth_gripper = depth_gripper.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 1, 84, 84)
                encoded_depth_gripper = self.depth_gripper_encoder(depth_gripper)
                encoded_depth_gripper = encoded_depth_gripper.reshape(b, s, -1)  # (batch, seq, 64)
                encoded_imgs = torch.cat([encoded_imgs, encoded_depth_gripper], dim=-1)

        if rgb_tactile is not None:
            b, s, c, h, w = rgb_tactile.shape
            rgb_tactile = rgb_tactile.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_tactile = self.tactile_encoder(rgb_tactile)
            encoded_tactile = encoded_tactile.reshape(b, s, -1)
            encoded_imgs = torch.cat([encoded_imgs, encoded_tactile], dim=-1)

        self.current_visual_embedding = encoded_imgs
        self.current_state_obs = state_obs  # type: ignore
        if self.proprio_encoder:
            state_obs_out = self.proprio_encoder(state_obs)
            perceptual_emb = torch.cat([encoded_imgs, state_obs_out], dim=-1)
            perceptual_emb_seg = torch.cat([encoded_imgs_seg, state_obs_out], dim=-1)
        else:
            perceptual_emb = encoded_imgs
            perceptual_emb_seg = encoded_imgs_seg
        
        
        return perceptual_emb, perceptual_emb_seg

    def state_reconstruction_loss(self):
        assert self.state_decoder is not None
        proprio_pred = self.state_decoder(self.current_visual_embedding)
        return mse_loss(self.current_state_obs, proprio_pred)
