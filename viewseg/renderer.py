# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import List, Optional, Tuple
import pdb

import torch
import torch.nn.functional as F
from pytorch3d.renderer import ImplicitRenderer
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points

from .nerf.implicit_function import NeuralRadianceField
from .implicit_function import SemanticNeuralRadianceField
from .nerf.raymarcher import EmissionAbsorptionNeRFRaymarcher
from .raysampler import NeRFRaysampler, ProbabilisticRaysampler
from .nerf.utils import calc_mse, calc_psnr, sample_images_at_mc_locs
from .raysampler import SphereRaysampler
from .raymarcher import MeshRaymarcher


class SemanticRadianceFieldRenderer(NeuralRadianceField):
    """
    Implements a renderer of a Neural Radiance Field.

    This class holds pointers to the fine and coarse renderer objects, which are
    instances of `pytorch3d.renderer.ImplicitRenderer`, and pointers to the
    neural networks representing the fine and coarse Neural Radiance Fields,
    which are instances of `NeuralRadianceField`.

    The rendering forward pass proceeds as follows:
        1) For a given input camera, rendering rays are generated with the
            `NeRFRaysampler` object of `self._renderer['coarse']`.
            In the training mode (`self.training==True`), the rays are a set
                of `n_rays_per_image` random 2D locations of the image grid.
            In the evaluation mode (`self.training==False`), the rays correspond
                to the full image grid. The rays are further split to
                `chunk_size_test`-sized chunks to prevent out-of-memory errors.
        2) For each ray point, the coarse `NeuralRadianceField` MLP is evaluated.
            The pointer to this MLP is stored in `self._implicit_function['coarse']`
        3) The coarse radiance field is rendered with the
            `EmissionAbsorptionNeRFRaymarcher` object of `self._renderer['coarse']`.
        4) The coarse raymarcher outputs a probability distribution that guides
            the importance raysampling of the fine rendering pass. The
            `ProbabilisticRaysampler` stored in `self._renderer['fine'].raysampler`
            implements the importance ray-sampling.
        5) Similar to 2) the fine MLP in `self._implicit_function['fine']`
            labels the ray points with occupancies and colors.
        6) self._renderer['fine'].raymarcher` generates the final fine render.
        7) The fine and coarse renders are compared to the ground truth input image
            with PSNR and MSE metrics.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        n_pts_per_ray: int,
        n_pts_per_ray_fine: int,
        n_rays_per_image: int,
        min_depth: float,
        max_depth: float,
        stratified: bool,
        stratified_test: bool,
        chunk_size_test: int,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        n_classes: int = 102,
        append_xyz: List[int] = (5,),
        density_noise_std: float = 0.0,
        scene_encoder = None,
        use_image_feats: bool = True,
        transform_to_source_view: bool = True,
        resnetfc: bool = True,
        ignore_index: int = -1,
        use_depth: bool = False,
        use_view_dirs: bool = True,
        sample_sphere: bool = False,
    ):
        """
        Args:
            image_size: The size of the rendered image (`[height, width]`).
            n_pts_per_ray: The number of points sampled along each ray for the
                coarse rendering pass.
            n_pts_per_ray_fine: The number of points sampled along each ray for the
                fine rendering pass.
            n_rays_per_image: Number of Monte Carlo ray samples when training
                (`self.training==True`).
            min_depth: The minimum depth of a sampled ray-point for the coarse rendering.
            max_depth: The maximum depth of a sampled ray-point for the coarse rendering.
            stratified: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during training (`self.training==True`).
            stratified_test: If `True`, stratifies (=randomly offsets) the depths
                of each ray point during evaluation (`self.training==False`).
            chunk_size_test: The number of rays in each chunk of image rays.
                Active only when `self.training==True`.
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
                Prior to evaluating the skip layers, the tensor which was input to MLP
                is appended to the skip layer input.
            density_noise_std: The standard deviation of the random normal noise
                added to the output of the occupancy MLP.
                Active only when `self.training==True`.
        """

        super().__init__()

        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()

        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()
        # raymarcher = MeshRaymarcher()

        # Parse out image dimensions.
        image_height, image_width = image_size

        for render_pass in ("coarse", "fine"):
            if render_pass == "coarse":
                # Initialize the coarse raysampler.
                if not sample_sphere:
                    raysampler = NeRFRaysampler(
                        n_pts_per_ray=n_pts_per_ray,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        stratified=stratified,
                        stratified_test=stratified_test,
                        n_rays_per_image=n_rays_per_image,
                        image_height=image_height,
                        image_width=image_width,
                    )
                else:
                    raysampler = SphereRaysampler(
                        n_pts_per_ray=n_pts_per_ray,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        stratified=stratified,
                        stratified_test=stratified_test,
                        n_rays_per_image=n_rays_per_image,
                        image_height=image_height,
                        image_width=image_width,
                    )
            elif render_pass == "fine":
                # Initialize the fine raysampler.
                raysampler = ProbabilisticRaysampler(
                    n_pts_per_ray=n_pts_per_ray_fine,
                    stratified=stratified,
                    stratified_test=stratified_test,
                )
            else:
                raise ValueError(f"No such rendering pass {render_pass}")

            # Initialize the fine/coarse renderer.
            self._renderer[render_pass] = ImplicitRenderer(
                raysampler=raysampler,
                raymarcher=raymarcher,
            )

            # Instantiate the fine/coarse NeuralRadianceField module.
            self._implicit_function[render_pass] = SemanticNeuralRadianceField(
                n_harmonic_functions_xyz=n_harmonic_functions_xyz,
                n_harmonic_functions_dir=n_harmonic_functions_dir,
                n_hidden_neurons_xyz=n_hidden_neurons_xyz,
                n_hidden_neurons_dir=n_hidden_neurons_dir,
                n_layers_xyz=n_layers_xyz,
                n_classes=n_classes,
                append_xyz=append_xyz,
                image_feature_dim=scene_encoder.latent_size if scene_encoder is not None else None,
                use_image_feats=use_image_feats, 
                transform_to_source_view=transform_to_source_view,
                resnetfc=resnetfc,
                use_depth=use_depth,
                use_view_dirs=use_view_dirs,
            )

        self.scene_encoder = scene_encoder
        self._n_classes = n_classes
        self._density_noise_std = density_noise_std
        self._chunk_size_test = chunk_size_test
        self._image_size = image_size
        self._ignore_index = ignore_index
        self._use_depth = use_depth

    def _process_ray_chunk(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        depth: torch.Tensor,
        semantic_label: torch.Tensor,
        source_camera: CamerasBase,
        source_image_feats: torch.Tensor,
        chunk_idx: int,
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
            chunk_idx: The index of the currently rendered ray chunk.
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.
        """
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            (features, weights), ray_bundle_out = self._renderer[renderer_pass](
                cameras=camera,
                volumetric_function=self._implicit_function[renderer_pass],
                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=(self._density_noise_std if self.training else 0.0),
                input_ray_bundle=coarse_ray_bundle,
                ray_weights=coarse_weights,
                camera_hash=camera_hash,
                # Source image info 
                source_image_feats=source_image_feats,
                source_cameras=source_camera,
                # target image info
                target_sem_label=semantic_label,
            )

            rgb = features[:, :, :3]
            pred_semantic = features[:, :, 3:(3 + self._n_classes)]
            pred_depth = features[:, :, (3 + self._n_classes):]

            if renderer_pass == "coarse":
                semantic_coarse = pred_semantic
                rgb_coarse = rgb
                if self._use_depth:
                    depth_coarse = pred_depth
                else:
                    depth_coarse = None
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights
                if image is not None:
                    # Sample the ground truth images at the xy locations of the
                    # rendering ray pixels.
                    rgb_gt = sample_images_at_mc_locs(
                        image[..., :3],
                        ray_bundle_out.xys,
                    )
                    semantic_gt = sample_images_at_mc_locs(
                        semantic_label[..., None].float(),
                        ray_bundle_out.xys,
                        mode="nearest",
                    )
                    semantic_gt = semantic_gt.long()
                    if self._use_depth:
                        depth_gt = sample_images_at_mc_locs(
                            depth[..., None].float(),
                            ray_bundle_out.xys,
                            mode="nearest",
                        )
                    else:
                        depth_gt = None
                else:
                    #rgb_gt = None
                    #semantic_gt = None
                    #depth_gt = None
                    rgb_gt = torch.zeros_like(rgb_coarse)
                    semantic_gt = torch.zeros_like(semantic_coarse)
                    depth_gt = None

            elif renderer_pass == "fine":
                semantic_fine = pred_semantic
                rgb_fine = rgb
                if self._use_depth:
                    depth_fine = pred_depth
                else:
                    depth_fine = None

            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        return {
            "rgb_fine": rgb_fine,
            "rgb_coarse": rgb_coarse,
            "rgb_gt": rgb_gt,
            "semantic_coarse": semantic_coarse,
            "semantic_fine": semantic_fine,
            "semantic_gt": semantic_gt,
            "depth_fine": depth_fine,
            "depth_coarse": depth_coarse,
            "depth_gt": depth_gt,
            # Store the coarse rays/weights only for visualization purposes.
            "coarse_ray_bundle": type(coarse_ray_bundle)(
                *[v.detach().cpu() for k, v in coarse_ray_bundle._asdict().items()]
            ),
            "coarse_weights": coarse_weights.detach().cpu(),
        }

    def forward(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        depth: torch.Tensor,
        sem_label: torch.Tensor,
        source_camera: CamerasBase,
        source_image: torch.Tensor,
        source_depth: torch.Tensor = None,
    ) -> Tuple[dict, dict]:
        """
        Performs the coarse and fine rendering passes of the radiance field
        from the viewpoint of the input `camera`.
        Afterwards, both renders are compared to the input ground truth `image`
        by evaluating the peak signal-to-noise ratio and the mean-squared error.

        The rendering result depends on the `self.training` flag:
            - In the training mode (`self.training==True`), the function renders
              a random subset of image rays (Monte Carlo rendering).
            - In evaluation mode (`self.training==False`), the function renders
              the full image. In order to prevent out-of-memory errors,
              when `self.training==False`, the rays are sampled and rendered
              in batches of size `chunksize`.

        Args:
            camera_hash: A unique identifier of a pre-cached camera.
                If `None`, the cache is not searched and the sampled rays are
                calculated from scratch.
            camera: A batch of cameras from which the scene is rendered.
            image: A batch of corresponding ground truth images of shape
                ('batch_size', ·, ·, 3).
            depth: target view depth
            sem_label: ('batch_size', ·, ·, num_classes).
            source_camera: source view camera
            source_image: source view rgb ('batch_size', ·, ·, 3)
            source_depth: source view depth
        Returns:
            out: `dict` containing the outputs of the rendering:
                `rgb_coarse`: The result of the coarse rendering pass.
                `rgb_fine`: The result of the fine rendering pass.
                `rgb_gt`: The corresponding ground-truth RGB values.

                The shape of `rgb_coarse`, `rgb_fine`, `rgb_gt` depends on the
                `self.training` flag:
                    If `==True`, all 3 tensors are of shape
                    `(batch_size, n_rays_per_image, 3)` and contain the result
                    of the Monte Carlo training rendering pass.
                    If `==False`, all 3 tensors are of shape
                    `(batch_size, image_size[0], image_size[1], 3)` and contain
                    the result of the full image rendering pass.
            metrics: `dict` containing the error metrics comparing the fine and
                coarse renders to the ground truth:
                `mse_coarse`: Mean-squared error between the coarse render and
                    the input `image`
                `mse_fine`: Mean-squared error between the fine render and
                    the input `image`
                `psnr_coarse`: Peak signal-to-noise ratio between the coarse render and
                    the input `image`
                `psnr_fine`: Peak signal-to-noise ratio between the fine render and
                    the input `image`
        """
        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                camera.R.shape[0],
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1
    

        # Extract features from the source image
        source_image_feats = None
        if self.scene_encoder is not None:
            source_image_feats = self.scene_encoder(source_image)

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                camera_hash,
                camera,
                image,
                depth,
                sem_label,
                source_camera,
                source_image_feats,
                chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        if not self.training:
            # For a full render pass concatenate the output chunks,
            # and reshape to image size.
            out = {
                k: torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 3)
                if chunk_outputs[0][k] is not None
                else None
                for k in ("rgb_fine", "rgb_coarse", "rgb_gt")
            }
            out['rgb_gt_source'] = source_image

            for k in ("semantic_fine", "semantic_coarse"):
                out[k] = torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, self._n_classes)

            for k in ("semantic_gt", ):
                out[k] = torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 1)

            # depth
            for k in ("depth_fine", "depth_coarse", "depth_gt"):
                if chunk_outputs[0][k] is not None:
                    out[k] = torch.cat(
                        [ch_o[k] for ch_o in chunk_outputs],
                        dim=1,
                    ).view(-1, *self._image_size, 1)
                else:
                    out[k] = None

        else:
            out = chunk_outputs[0]

        # Calc the error metrics.
        metrics = {}
        if image is not None:
            for render_pass in ("coarse", "fine"):
                # rgb loss
                for metric_name, metric_fun in zip(
                    ("mse", "psnr"), (calc_mse, calc_psnr)
                ):
                    metrics[f"{metric_name}_{render_pass}"] = metric_fun(
                        out["rgb_" + render_pass][..., :3],
                        out["rgb_gt"][..., :3],
                    )

                # semantic loss
                if self.training:
                    loss_semantic = F.cross_entropy(out['semantic_' + render_pass].permute(0, 2, 1), out['semantic_gt'][:, :, 0], ignore_index=self._ignore_index)
                else:
                    loss_semantic = F.cross_entropy(out['semantic_' + render_pass].permute(0, 3, 1, 2), out['semantic_gt'][:, :, :, 0], ignore_index=self._ignore_index)
                metrics["semantic_" + render_pass] = loss_semantic

                # depth loss
                if self._use_depth:
                    invalid_pixels = torch.logical_or(torch.isnan(out['depth_gt']), out['depth_gt'] > 20.0)
                    valid_mask = torch.logical_not(invalid_pixels)
                    loss_depth = (out["depth_" + render_pass][valid_mask] - out["depth_gt"][valid_mask]) ** 2
                    #loss_depth = loss_depth[valid_mask]
                    loss_depth = loss_depth.mean()
                    if torch.isnan(loss_depth): # all pixels could be invalid
                        loss_depth = 0.0
                    metrics["depth_" + render_pass] = loss_depth
                else:
                    metrics["depth_" + render_pass] = 0.0

        return out, metrics

    def forward_sphere(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        depth: torch.Tensor,
        sem_label: torch.Tensor,
        source_camera: CamerasBase,
        source_image: torch.Tensor,
        source_depth: torch.Tensor = None,
    ) -> Tuple[dict, dict]:
        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                camera.R.shape[0],
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1
    

        # Extract features from the source image
        source_image_feats = None
        if self.scene_encoder is not None:
            source_image_feats = self.scene_encoder(source_image)

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                camera_hash,
                camera,
                image,
                depth,
                sem_label,
                source_camera,
                source_image_feats,
                chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]

        out = {}

        rgb_fine = []
        semantic_fine = []
        depth_fine = []
        points = []
        for chunk_idx in range(n_chunks):
            chunk_rgb_fine = chunk_outputs[chunk_idx]['rgb_fine']
            chunk_semantic_fine = chunk_outputs[chunk_idx]['semantic_fine']
            chunk_depth_fine = chunk_outputs[chunk_idx]['depth_fine']
            chunk_ray_bundle = chunk_outputs[chunk_idx]['coarse_ray_bundle']
            chunk_points = chunk_ray_bundle.origins.cuda() + chunk_depth_fine * chunk_ray_bundle.directions.cuda()

            rgb_fine.append(chunk_rgb_fine)
            semantic_fine.append(chunk_semantic_fine)
            depth_fine.append(chunk_depth_fine)
            points.append(chunk_points)

        
        rgb_fine = torch.cat(rgb_fine, dim=1).view(-1, *self._image_size, 3)
        semantic_fine = torch.cat(semantic_fine, dim=1).view(-1, *self._image_size, self._n_classes)
        depth_fine = torch.cat(depth_fine, dim=1).view(-1, *self._image_size, 1)
        points = torch.cat(points, dim=1)

        # rgb_fine = chunk_outputs[0]['rgb_fine']
        # semantic_fine = chunk_outputs[0]['semantic_fine']
        # depth_fine = chunk_outputs[0]['depth_fine']
        # ray_bundle = chunk_outputs[0]['coarse_ray_bundle']
        # points = ray_bundle.origins.cuda() + depth_fine * ray_bundle.directions.cuda()
        
        # rgb_fine = rgb_fine.view(1, 60, 100, 3)
        # semantic_fine = semantic_fine.view(1, 60, 100, -1)
        # depth_fine = depth_fine.view(1, 60, 100, 1)

        # rgb_fine = rgb_fine.view(1, *self._image_size, 3)
        # semantic_fine = semantic_fine.view(1, *self._image_size, -1)
        # depth_fine = depth_fine.view(1, *self._image_size, 1)

        out['rgb_fine'] = rgb_fine
        out['semantic_fine'] = semantic_fine
        out['depth_fine'] = depth_fine
        out['points'] = points

        return out, None

    def _process_ray_chunk_mesh(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        depth: torch.Tensor,
        semantic_label: torch.Tensor,
        source_camera: CamerasBase,
        source_image_feats: torch.Tensor,
        chunk_idx: int,
    ) -> dict:
        """
        Samples and renders a chunk of rays.

        for mesh instead of 2d rendering features
        """
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # fine pass only
        for renderer_pass in ("coarse", ): # "fine"):
            (features, weights, rays_features, rays_densities), ray_bundle_out = self._renderer[renderer_pass](
                cameras=camera,
                volumetric_function=self._implicit_function[renderer_pass],
                chunksize=self._chunk_size_test,
                chunk_idx=chunk_idx,
                density_noise_std=(self._density_noise_std if self.training else 0.0),
                input_ray_bundle=coarse_ray_bundle,
                ray_weights=coarse_weights,
                camera_hash=camera_hash,
                # Source image info 
                source_image_feats=source_image_feats,
                source_cameras=source_camera,
                # target image info
                target_sem_label=semantic_label,
            )

            rgb = features[:, :, :3]
            pred_semantic = features[:, :, 3:(3 + self._n_classes)]
            pred_depth = features[:, :, (3 + self._n_classes):]

            rays_rgb = rays_features[:, :, :, :3]
            rays_semantic = rays_features[:, :, :, 3:(3 + self._n_classes)]

            if renderer_pass == "coarse":
                semantic_coarse = pred_semantic
                rgb_coarse = rgb
                if self._use_depth:
                    depth_coarse = pred_depth
                else:
                    depth_coarse = None
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights
                if image is not None:
                    # Sample the ground truth images at the xy locations of the
                    # rendering ray pixels.
                    rgb_gt = sample_images_at_mc_locs(
                        image[..., :3],
                        ray_bundle_out.xys,
                    )
                    semantic_gt = sample_images_at_mc_locs(
                        semantic_label[..., None].float(),
                        ray_bundle_out.xys,
                        mode="nearest",
                    )
                    semantic_gt = semantic_gt.long()
                    if self._use_depth:
                        depth_gt = sample_images_at_mc_locs(
                            depth[..., None].float(),
                            ray_bundle_out.xys,
                            mode="nearest",
                        )
                    else:
                        depth_gt = None
                else:
                    rgb_gt = torch.zeros_like(rgb_coarse)
                    semantic_gt = torch.zeros_like(semantic_coarse)
                    depth_gt = None

            elif renderer_pass == "fine":
                semantic_fine = pred_semantic
                rgb_fine = rgb
                if self._use_depth:
                    depth_fine = pred_depth
                else:
                    depth_fine = None

            else:
                raise ValueError(f"No such rendering pass {renderer_pass}")

        return {
            #"depth_fine": depth_fine,
            "depth_coarse": depth_coarse,
            "semantic_coarse": semantic_coarse,
            "rays_densities": rays_densities,
            "rays_rgb": rays_rgb,
            "rays_semantic": rays_semantic,
            #"rays_depth": depth_fine,
            # Store the coarse rays/weights only for visualization purposes.
            "ray_bundle": type(ray_bundle_out)(
                *[v.detach().cpu() for k, v in ray_bundle_out._asdict().items()]
            ),
        }

    def forward_mesh(
        self,
        camera_hash: Optional[str],
        camera: CamerasBase,
        image: torch.Tensor,
        depth: torch.Tensor,
        sem_label: torch.Tensor,
        source_camera: CamerasBase,
        source_image: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """
        Generates mesh instead of 2d rendering features.
        """
        if not self.training:
            # Full evaluation pass.
            n_chunks = self._renderer["coarse"].raysampler.get_n_chunks(
                self._chunk_size_test,
                camera.R.shape[0],
            )
        else:
            # MonteCarlo ray sampling.
            n_chunks = 1
            raise ValueError("forward_mesh is eval only")
    

        # Extract features from the source image
        source_image_feats = None
        if self.scene_encoder is not None:
            source_image_feats = self.scene_encoder(source_image)

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk_mesh(
                camera_hash,
                camera,
                image,
                depth,
                sem_label,
                source_camera,
                source_image_feats,
                chunk_idx,
            )
            for chunk_idx in range(n_chunks)
        ]


        # For a full render pass concatenate the output chunks,
        # and reshape to image size.
        out = {
            k: torch.cat(
                [ch_o[k].cpu() for ch_o in chunk_outputs],
                dim=1,
            ).view(-1, *self._image_size, 64)
            if chunk_outputs[0][k] is not None
            else None
            for k in ("rays_densities", )
        }

        out['rays_semantic'] = torch.cat(
            [ch_o['rays_semantic'].argmax(dim=-1).cpu() for ch_o in chunk_outputs],
            dim=1,
        ).view(-1, *self._image_size, 64)

        out['rays_points'] = torch.cat(
            [ray_bundle_to_ray_points(ch_o['ray_bundle']) for ch_o in chunk_outputs],
            dim=1,
        ).view(-1, *self._image_size, 64, 3)

        out['semantic_coarse'] = torch.cat(
            [ch_o['semantic_coarse'] for ch_o in chunk_outputs],
            dim=1,
        ).view(-1, *self._image_size, self._n_classes)

        for k in ("depth_coarse", ):
            if chunk_outputs[0][k] is not None:
                out[k] = torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 1)
            else:
                out[k] = None

        # threshold and convert voxel grids to pcd
        
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(out['rays_densities'].numpy().ravel(), log=True)
        # plt.savefig('/private/home/syqian/panonerf/panonerf/hist.png')
        # pdb.set_trace()

        pcd_mask = out['rays_densities'] > 4.0
        #wall_mask = torch.logical_and(out['rays_densities'] > 5.0, out['rays_semantic'] == 92)
        #pcd_mask = torch.logical_or(pcd_mask, wall_mask)
        #blind_mask = torch.logical_and(out['rays_densities'] > 5.0, out['rays_semantic'] == 11)
        #pcd_mask = torch.logical_or(pcd_mask, blind_mask)
        #pcd_mask = out['rays_densities'] > 5.0
        pcd_points = out['rays_points'][pcd_mask]
        pcd_semantic = out['rays_semantic'][pcd_mask]

        # depth
        """
        for k in ("depth_fine",):
            if chunk_outputs[0][k] is not None:
                depth_fine = torch.cat(
                    [ch_o[k] for ch_o in chunk_outputs],
                    dim=1,
                ).view(-1, *self._image_size, 1)
            else:
                depth_fine = None
        """

        out = {
            'pcd_points': pcd_points,
            'pcd_semantic': pcd_semantic,
            'semantic_coarse': out['semantic_coarse'],
            #"depth_fine": depth_fine,
        }

        return out, None
