from ray import tune

MNIST = dict(
    name='mnist_base',
    dataset='MNIST',
    batch_size=1024,
    image_size=28,
    seed=42,
    model_args=dict(
        image_size=28,
        n_channels=1,
        num_patch_side=7,
        hidden_dim=64,
        levels=2,
        iters=tune.grid_search([5,10]),
        contributions=tune.grid_search([[0.25,0.25,0.25,0.25],[0.15,0.35,0.15,0.35]]),
        recon_coeff=1.0,
        local_coeff=0.1,
        local_consensus_radius=0,
        optimizer_args=dict(
            lr=0.05,
            decay=5e-4,
            steps_per_epoch=45000 // 1024,
            epochs=2000),
        overlapping_embedding=tune.grid_search([True,False]),
        reconstruction_end=tune.grid_search([True, False]),
        latent_reconstruction=tune.grid_search([True, False]),
        location_embedding=tune.grid_search([True, False]),
        add_embedding=False
    ),
    trainer_args=dict(
        accelerator='gpu',
        gpus=None,
        strategy='dp',
        max_epochs=20,
        reload_dataloaders_every_n_epochs=1
    )
)

CONFIGS = dict(
    mnist_base=MNIST
)

