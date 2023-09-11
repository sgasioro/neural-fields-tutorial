@torch.no_grad()
def make_image(scene, device='cpu', nb_rays=int(1e9), batch_size=int(50e6), quantum_efficiency=True, max_iterations=2,
               add_poisson_noise=True, lookup_table=None, show_progress=True, destructive_readout=True):
    
    progress_bar = tqdm if show_progress else lambda x: x
    nb_rays_left_to_sample = nb_rays
    for _ in progress_bar(range(int(np.ceil(nb_rays / batch_size)))):
        rays = scene.light_source.sample_rays(min(batch_size, nb_rays_left_to_sample), device=device)
        
        optics.forward_ray_tracing(rays, scene, max_iterations=max_iterations, quantum_efficiency=quantum_efficiency)
        nb_rays_left_to_sample -= batch_size

        del rays
        torch.cuda.empty_cache()

    return scene.objects[0].readout(add_poisson_noise=add_poisson_noise, destructive_readout=destructive_readout)