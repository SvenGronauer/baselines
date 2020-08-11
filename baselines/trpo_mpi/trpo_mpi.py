from baselines.common import explained_variance, zipsame, dataset
from research.common import loggers
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
import time
import os.path as osp
from baselines.common import colorize
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.models import get_network_builder
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.policies import PolicyWithValue
from baselines.common.vec_env.vec_env import VecEnv
from contextlib import contextmanager

# try:
#     from mpi4py import MPI
# except ImportError:
#     MPI = None

MPI = None  # disable MPI by default -> use multi-processes in parallel instead


def traj_segment_generator(pi, env, horizon, training):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()
    if not isinstance(env, VecEnv):
      ob = np.expand_dims(ob, axis=0)

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ob = tf.constant(ob)
        ac, vpred, _, _ = pi.step(ob, training=training)
        ac = ac.numpy()
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred, _, _ = pi.step(ob, training=training)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred.numpy()
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        # print('Trajectory Rew:', rew)
        # time.sleep(1)
        # if new[0]:
        #     print(i, rew)
        if not isinstance(env, VecEnv):
          ob = np.expand_dims(ob, axis=0)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            # print('i:', i)
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            # print('cur_ep_ret:', cur_ep_ret)
            # print('cur_ep_len:', cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            if not isinstance(env, VecEnv):
              ob = np.expand_dims(ob, axis=0)
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(*,
        network,
        env,
        eval_env,
        total_timesteps,
        logger_kwargs,
        nbatch,  # what to train on
        max_kl,
        cg_iters,
        gamma,
        lam,  # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    nbatch                  timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''
    verbose = logger_kwargs['verbose']

    # Set up logger and save configuration
    params = locals()  # get before logger instance to avoid unnecessary prints
    logger = loggers.EpochLogger(**logger_kwargs)
    params.pop('env')  # make config.json more readable
    logger.save_config(params)

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    if isinstance(network, str):
        network = get_network_builder(network)(**network_kwargs)

    with tf.name_scope("pi"):
        pi_policy_network = network(ob_space.shape)
        pi_value_network = network(ob_space.shape)
        pi = PolicyWithValue(ac_space, pi_policy_network, pi_value_network)
    with tf.name_scope("oldpi"):
        old_pi_policy_network = network(ob_space.shape)
        old_pi_value_network = network(ob_space.shape)
        oldpi = PolicyWithValue(ac_space, old_pi_policy_network, old_pi_value_network)

    pi_var_list = pi_policy_network.trainable_variables + list(pi.pdtype.trainable_variables)
    old_pi_var_list = old_pi_policy_network.trainable_variables + list(oldpi.pdtype.trainable_variables)
    vf_var_list = pi_value_network.trainable_variables + pi.value_fc.trainable_variables
    old_vf_var_list = old_pi_value_network.trainable_variables + oldpi.value_fc.trainable_variables

    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=pi)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(pi_var_list)
    set_from_flat = U.SetFromFlat(pi_var_list)
    loss_names = ["OptimGain", "KL", "EntLoss", "LossPi", "Entropy"]
    shapes = [var.get_shape().as_list() for var in pi_var_list]


    def assign_old_eq_new():
        for pi_var, old_pi_var in zip(pi_var_list, old_pi_var_list):
            old_pi_var.assign(pi_var)
        for vf_var, old_vf_var in zip(vf_var_list, old_vf_var_list):
            old_vf_var.assign(vf_var)

    @tf.function
    def compute_lossandgrad(ob, ac, atarg, training):
        with tf.GradientTape() as tape:
            old_policy_latent = oldpi.policy_network(ob, training=training)
            old_pd, _ = oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = pi.policy_network(ob, training=training)
            pd, _ = pi.pdtype.pdfromlatent(policy_latent)
            kloldnew = old_pd.kl(pd)
            ent = pd.entropy()
            meankl = tf.reduce_mean(kloldnew)
            meanent = tf.reduce_mean(ent)
            entbonus = ent_coef * meanent
            ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
            surrgain = tf.reduce_mean(ratio * atarg)
            optimgain = surrgain + entbonus
            losses = [optimgain, meankl, entbonus, surrgain, meanent]
        gradients = tape.gradient(optimgain, pi_var_list)
        return losses + [U.flatgrad(gradients, pi_var_list)]

    @tf.function
    def compute_losses(ob, ac, atarg, training=True):
        old_policy_latent = oldpi.policy_network(ob, training=training)
        old_pd, _ = oldpi.pdtype.pdfromlatent(old_policy_latent)
        policy_latent = pi.policy_network(ob, training=training)
        pd, _ = pi.pdtype.pdfromlatent(policy_latent)
        kloldnew = old_pd.kl(pd)
        ent = pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = ent_coef * meanent
        ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
        surrgain = tf.reduce_mean(ratio * atarg)
        optimgain = surrgain + entbonus
        losses = [optimgain, meankl, entbonus, surrgain, meanent]
        return losses

    #ob shape should be [batch_size, ob_dim], merged nenv
    #ret shape should be [batch_size]
    @tf.function
    def compute_vflossandgrad(ob, ret, training=True):
        with tf.GradientTape() as tape:
            pi_vf = pi.value(ob, training=training)
            vferr = tf.reduce_mean(tf.square(pi_vf - ret))
        return U.flatgrad(tape.gradient(vferr, vf_var_list), vf_var_list)

    @tf.function
    def compute_fvp(flat_tangent, ob, ac, atarg, training=True):
        with tf.GradientTape() as outter_tape:
            with tf.GradientTape() as inner_tape:
                old_policy_latent = oldpi.policy_network(ob, training=training)
                old_pd, _ = oldpi.pdtype.pdfromlatent(old_policy_latent)
                policy_latent = pi.policy_network(ob, training=training)
                pd, _ = pi.pdtype.pdfromlatent(policy_latent)
                kloldnew = old_pd.kl(pd)
                meankl = tf.reduce_mean(kloldnew)
            klgrads = inner_tape.gradient(meankl, pi_var_list)
            start = 0
            tangents = []
            for shape in shapes:
                sz = U.intprod(shape)
                tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
                start += sz
            gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])
        hessians_products = outter_tape.gradient(gvp, pi_var_list)
        fvp = U.flatgrad(hessians_products, pi_var_list)
        return fvp

    @contextmanager
    def timed(msg, verbose):
        if rank == 0:
            print(colorize(msg, color='magenta')) if verbose else None
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta')) if verbose else None
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
        else:
            out = np.copy(x)

        return out

    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)

    set_from_flat(th_init)
    vfadam.sync()
    # print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, nbatch, training=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far) if verbose else None

        with timed("sampling", verbose):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        ob = sf01(ob)
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = ob, ac, atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            fvp = allmean(compute_fvp(p, *fvpargs).numpy()) + cg_damping * p
            # print('Norm Fvp:', np.linalg.norm(fvp))
            return fvp

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad", verbose):
            *lossbefore, g = compute_lossandgrad(*args, training=True)
        lossbefore = allmean(np.array(lossbefore))
        g = g.numpy()
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating") if verbose else None
        else:
            with timed("cg", verbose):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters,
                             verbose=False)
            assert np.isfinite(stepdir).all()
            # print('Norm stepdir:', np.linalg.norm(stepdir))
            # logger.log_tabular('TRPO/StepDirection', np.linalg.norm(stepdir))
            # logger.log_tabular('FullStepNorm', np.linalg.norm(stepdir))
            shs = stepdir.dot(fisher_vector_product(stepdir))
            # logger.log_tabular('TRPO/xHx', shs)
            lm = np.sqrt(shs / 2 * max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            # print('Norm Fullstep:', )
            logger.store(FullStepNorm=np.linalg.norm(fullstep),
                         xHx=shs,
                         gradient_norm=np.linalg.norm(g),
                         cg_x=np.linalg.norm(stepdir))
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for k in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve)) if verbose else None
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!") if verbose else None
                elif kl > max_kl * 1.5:
                    pass
                    # logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    pass
                    # logger.log("surrogate didn't improve. shrinking step.")
                else:
                    # logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("Couldn't compute a good step") if verbose else None
                set_from_flat(thbefore)
            logger.store(AcceptanceStep=k)

            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        # log the mean of all losses
        logger.store(**dict(zip(loss_names, meanlosses)))

        with timed("vf", verbose):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    mbob = sf01(mbob)
                    g = allmean(compute_vflossandgrad(mbob, mbret).numpy())
                    vfadam.update(g, vf_stepsize)
        explaned_var = explained_variance(vpredbefore, tdlamret)
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.store(
            EpLen=lenbuffer,
            EpRet=rewbuffer,
            EpThisIter=len(lens)
        )
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        if rank == 0:
            logger.log_tabular('Epoch', iters_so_far)
            logger.log_tabular('EpRet', min_and_max=True, std=True)
            logger.log_tabular('EpLen', min_and_max=True)
            logger.log_tabular('ExplainedVariance', explaned_var)
            logger.log_tabular('FullStepNorm')
            logger.log_tabular('gradient_norm')
            logger.log_tabular('xHx')
            logger.log_tabular('cg_x')
            logger.log_tabular('OptimGain')
            logger.log_tabular('KL')
            logger.log_tabular('EntLoss')
            logger.log_tabular('LossPi')
            logger.log_tabular('Entropy')
            logger.log_tabular('AcceptanceStep')
            logger.log_tabular('Time', time.time() - tstart)
            logger.log_tabular('TotalEnvSteps', timesteps_so_far)
            logger.log_tabular('FPS', timesteps_so_far / (time.time() - tstart))
            logger.dump_tabular()

    return pi


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
