from baselines.common.models import mlp  #, cnn_small


# def atari():
#     return dict(
#         network = cnn_small(),
#         timesteps_per_batch=512,
#         max_kl=0.001,
#         cg_iters=10,
#         cg_damping=1e-3,
#         gamma=0.98,
#         lam=1.0,
#         vf_iters=3,
#         vf_stepsize=1e-4,
#         entcoeff=0.00,
#     )

def mujoco():
    return dict(
        network = mlp(num_hidden=32, num_layers=2),
        timesteps_per_batch=1024,
        max_kl=0.01,
        cg_iters=10,
        cg_damping=0.1,
        gamma=0.99,
        lam=0.98,
        vf_iters=5,
        vf_stepsize=1e-3,
        normalize_observations=True,
    )


def gym_locomotion_envs():
    return dict(
        total_timesteps=int(10e6),
        nbatch=2000*16,
        nminibatches=32,
        lam=0.95,
        cg_damping=0.1,
        cg_iters=10,
        gamma=0.99,
        max_kl=0.01,
        noptepochs=20,
        # num_timesteps=str(int(10e6)),  # default is 10e6
        log_interval=1,
        ent_coef=0.0,
        reward_scale=1.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='separate',
        network='mlp',
        vf_stepsize=3e-4,
        vf_iters=3,
        network_kwargs={
            'activation': 'tanh',
            'dropout': False,
            'dropout_rate': 0.5,
            'norm_apply': False,
            'norm_coefficient': 1.0e-4,
            'norm_type': 'L2',
            'num_hidden': 64,
            'num_layers': 2
        }
    )
