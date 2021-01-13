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
        cg_damping=0.1,
        cg_iters=10,
        ent_coef=0.0,
        gamma=0.99,
        lam=0.95,
        max_kl=0.01,
        network='mlp',
        network_kwargs={
            'activation': 'tanh',
            'dropout': False,
            'dropout_rate': 0.5,
            'norm_apply': False,
            'norm_coefficient': 1.0e-4,
            'norm_type': 'L2',
            'num_hidden': 64,
            'num_layers': 2
        },
        reward_scale=1.0,  # necessary for setup_baselines.py
        value_network='separate',
        vf_stepsize=3e-4,
        vf_iters=5
    )


def gym_manipulator_envs():
    """ Default hyper-parameters for PyBullet's manipulation tasks,
    e.g. ReacherBulletEnv-0, PusherBulletEnv-v0"""
    return gym_locomotion_envs()


def bullet():
    """ Default hyper-parameters for PyBullet Envs such as KukaBulletEnv-v0."""
    return gym_locomotion_envs()
