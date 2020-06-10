def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )


def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )


def retro():
    return atari()


def gym_locomotion_envs():
    return dict(
        total_timesteps=int(10e6),  #todo back to 10e6
        nsteps=2000*16,
        nminibatches=16,
        lam=0.95,
        gamma=0.99,
        noptepochs=20,  # policy_iters = noptepochs * nminibatches
        num_env=1,
        # num_timesteps=str(int(10e6)),  # default is 10e6
        log_interval=1,
        ent_coef=0.0,
        reward_scale=1.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='separate',
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
        }
    )
