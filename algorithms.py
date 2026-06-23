import math
import random
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple

import numpy as np
import torch
from torch.functional import norm

from base_optimizers import configure_base_optimizer
from tasks.api import Task
from topologies import configure_topology
from utils.communication import (
    MultiTopologyGossipMechanism,
    MultiTopologyRelayMechanism,
    get_rank,
    get_world_size,
    isend,
    num_bytes,
    pack,
    recv,
    unpack,
)
from utils.timer import Timer


class TrainStats(NamedTuple):
    step: int
    bytes_sent: int


class BatchStats(NamedTuple):
    loss: float


Parameters = List[torch.Tensor]


State = List[torch.Tensor]


def train(
    config: Dict[str, Any], task: Task, timer: Timer
) -> Iterable[Tuple[TrainStats, BatchStats, Parameters, State]]:
    if config["algorithm"] == "all-reduce":
        yield from allreduce(config, task, timer)
    elif config["algorithm"] == "gossip":
        yield from gossip(config, task, timer)
    elif config["algorithm"] == "d2":
        yield from d2(config, task, timer)
    elif config["algorithm"] == "edm":
        yield from edm(config, task, timer)
    elif config["algorithm"] == "caedm":
        yield from caedm(config, task, timer)
    elif config["algorithm"] == "gradient-tracking":
        yield from gradient_tracking(config, task, timer)
    elif config["algorithm"] == "quasi-global-momentum":
        yield from quasi_global_momentum(config, task, timer)
    elif config["algorithm"] == "dmsgd_gt":
        yield from dmsgd_gt(config, task, timer)
    elif config["algorithm"] == "dmsgt_hb":
        yield from dmsgt_hb(config, task, timer)
    elif config["algorithm"] == "dsgt_hb":
        yield from dsgt_hb(config, task, timer)
    elif config["algorithm"] == "dsmt":
        yield from dsmt(config, task, timer)
    elif config["algorithm"] == "push_sum":
        yield from push_sum(config, task, timer)
    elif config["algorithm"] == "decent_lam":
        yield from decent_lam(config, task, timer)
    else:
        raise ValueError("Unsupported algorithm {}".format(config["algorithm"]))


def get_gossip_weight(config: Dict[str, Any]):
    return config.get("gossip_weight", None)



def allreduce(config, task: Task, timer: Timer):
    assert config["topology"] == "fully-connected"

    bytes_sent = 0
    last_loss = None

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    gradients = 0.0

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        if config["overlap_communication"] and step > 0:
            with timer("communication.send"):
                buffer, shapes = pack(gradients)
                comm_handle = torch.distributed.all_reduce(buffer, async_op=True)
                bytes_sent += num_bytes(buffer)

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        if config["overlap_communication"] and step > 0:
            with timer("communication.recv"):
                comm_handle.wait()
                buffer /= get_world_size()
                avg_gradients = unpack(buffer, shapes)

        if not config["overlap_communication"] or step > 0:
            with timer("local_update"):
                base_optimizer.step(
                    parameters,
                    avg_gradients if config["overlap_communication"] else gradients,
                    base_optimizer_state,
                    lr=config["learning_rate"] * learning_rate_schedule(config, step),
                )

        if not config["overlap_communication"]:
            with timer("communication"):
                buffer, shapes = pack(parameters)
                torch.distributed.all_reduce(buffer)
                buffer /= get_world_size()
                parameters = unpack(buffer, shapes)
                bytes_sent += num_bytes(buffer)


def gossip(config, task: Task, timer: Timer):
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        if config["overlap_communication"]:
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            del buffer

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )

        with timer("communication"):
            buffer, shapes = pack(parameters)
            if not config["overlap_communication"]:
                gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)


def dmsgd_gt(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )
    correction_gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    correction = [torch.zeros_like(p) for p in parameters]
    m_t = [torch.zeros_like(p) for p in parameters]

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent + correction_gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("communication.correction"):
            # buffer, shapes = pack(correction)
            # print(get_rank(),buffer)
            for m, g, c in zip(m_t, gradients,correction):
                prev_m = m.clone()
                m.mul_(config["momentum"]).add_(g)
                c.add_(m-prev_m)
            buffer, shapes = pack(correction)
            # print(get_rank(),buffer*1e4)
            # if(get_rank()==0):
            #     print('***********')
            #     print(0, buffer)
            #     print('***********')
            correction_gossip.send(buffer)
            correction_gossip.gossip_update(buffer)
            correction = unpack(buffer, shapes)
            # if(get_rank()==0):
            #     print('***********')
            #     print(0, buffer*1e4)
            #     print('***********')


        with timer("local_update"):
            # if(get_rank()==0):
            #     print('****************')
            #     buffer, shapes = pack(parameters)
            #     print(0, buffer)
            #     buffer, shapes = pack(correction)
            #     print(0, buffer)
            #     print('****************')
            # for p, c in zip(parameters, correction):
            #     p.add_(-c,alpha=config["learning_rate"] * learning_rate_schedule(config, step))
            base_optimizer.step(
                parameters,
                correction,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            # if(get_rank()==0):
            #     print('***********')
            #     buffer, shapes = pack(parameters)
            #     print(0, buffer)
            #     print('***********')

        with timer("communication.parameters"):
            buffer, shapes = pack(parameters)
            # print(get_rank(),buffer)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            # if(get_rank()==0):
            #     print('***********')
            #     print(0, buffer)
            #     print('***********')
            parameters = unpack(buffer, shapes)
            

def dmsgt_hb(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )
    correction_gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    correction = [torch.zeros_like(p) for p in parameters]
    momentum = [torch.zeros_like(p) for p in parameters]

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent + correction_gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )
            correction = [b + u for b, u in zip(correction, gradients)]
        
        with timer("communication.correction"):
            buffer, shapes = pack(correction)
            correction_gossip.send(buffer)
            correction_gossip.gossip_update(buffer)
            correction = unpack(buffer, shapes)
        
        with timer("local_update"):
            for u, c in zip(momentum, correction):
                u.mul_(config["momentum"])
                u.add_(c, alpha = (1-config["momentum"]))
            base_optimizer.step(
                parameters,
                momentum,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )

        with timer("communication.parameters"):                 # x_{t+1} = Wx_{t+1/2}
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)
            correction = [b - u for b, u in zip(correction, gradients)]
            

def dsgt_hb(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    assert config["base_optimizer"] == "SGD"
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )
    correction_gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    tracking = None
    momentum = [torch.zeros_like(p) for p in parameters]

    data_iter = iter(
        task.data.iterator(
            batch_size=config["batch_size"],
            shuffle=True,
            ref_num_data=task.mean_num_data_per_worker,
        )
    )

    step, batch = next(data_iter)
    timer.epoch = step
    yield (
        TrainStats(step, gossip.bytes_sent + correction_gossip.bytes_sent),
        BatchStats(loss=last_loss),
        parameters,
        state,
    )

    with timer("compute_grad"):
        last_loss, gradients, state = task.loss_and_gradient(
            parameters, state, batch
        )
        tracking = [g.clone() for g in gradients]

    while True:
        with timer("local_update"):
            for u, s in zip(momentum, tracking):
                u.mul_(config["momentum"])
                u.add_(s, alpha=1 - config["momentum"])
            base_optimizer.step(
                parameters,
                momentum,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )

        with timer("communication.parameters"):
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)

        next_step, next_batch = next(data_iter)
        timer.epoch = next_step

        with timer("compute_grad"):
            last_loss, next_gradients, state = task.loss_and_gradient(
                parameters, state, next_batch
            )

        with timer("communication.correction"):
            buffer, shapes = pack(
                [s + g_next - g for s, g_next, g in zip(tracking, next_gradients, gradients)]
            )
            correction_gossip.send(buffer)
            correction_gossip.gossip_update(buffer)
            tracking = unpack(buffer, shapes)

        step = next_step
        batch = next_batch
        gradients = next_gradients

        yield (
            TrainStats(step, gossip.bytes_sent + correction_gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )


def dsmt(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    assert config["base_optimizer"] == "SGD"
    last_loss = None

    topology = configure_topology(config)
    assert not isinstance(topology, list)
    eta_w = config.get("lca_eta_w", config.get("eta_w", None))
    if eta_w is None:
        if topology.num_workers <= 1:
            lambda_w = 0.0
        else:
            w = topology.gossip_matrix(get_gossip_weight(config))
            eigenvalues = np.linalg.eigvalsh(w.detach().cpu().numpy())
            lambda_w = sorted(np.abs(eigenvalues).tolist())[-2]
        eta_w = 1.0 / (1.0 + math.sqrt(max(0.0, 1.0 - lambda_w * lambda_w)))

    if topology.num_workers <= 1:
        gossip = None
        tracking_gossip = None
    else:
        gossip = MultiTopologyGossipMechanism(
            topology,
            gossip_matrix=get_gossip_weight(config),
            message_drop_prob=config["simulated_dropped_message_probability"],
        )
        tracking_gossip = MultiTopologyGossipMechanism(
            topology,
            gossip_matrix=get_gossip_weight(config),
            message_drop_prob=config["simulated_dropped_message_probability"],
        )

    def lca_update(gossip_mechanism, values, lower_values):
        buffer, shapes = pack(values)
        if gossip_mechanism is not None:
            gossip_mechanism.send(buffer)
            gossip_mechanism.gossip_update(buffer)
        lower_buffer, _ = pack(lower_values)
        buffer.mul_(1 + eta_w).add_(lower_buffer, alpha=-eta_w)
        return unpack(buffer, shapes)

    def bytes_sent():
        if gossip is None:
            return 0
        return gossip.bytes_sent + tracking_gossip.bytes_sent

    parameters, state = task.initialize(seed=config["seed"])
    lower_parameters = [p.clone() for p in parameters]

    data_iter = iter(
        task.data.iterator(
            batch_size=config["batch_size"],
            shuffle=True,
            ref_num_data=task.mean_num_data_per_worker,
        )
    )

    step, batch = next(data_iter)
    timer.epoch = step
    yield (
        TrainStats(step, bytes_sent()),
        BatchStats(loss=last_loss),
        parameters,
        state,
    )

    with timer("compute_grad"):
        last_loss, gradients, state = task.loss_and_gradient(
            parameters, state, batch
        )

    beta = config["momentum"]
    momentum = [g.clone().mul_(1 - beta) for g in gradients]
    tracking = [z.clone() for z in momentum]
    tracking_lower = [z.clone() for z in momentum]

    while True:
        lr = config["learning_rate"] * learning_rate_schedule(config, step)

        with timer("local_update"):
            parameters_half = [
                p.clone().add_(y, alpha=-lr)
                for p, y in zip(parameters, tracking)
            ]
            lower_half = [
                p_l.clone().add_(y, alpha=-lr)
                for p_l, y in zip(lower_parameters, tracking)
            ]

        with timer("communication.parameters"):
            parameters = lca_update(gossip, parameters_half, lower_half)
            lower_parameters = parameters_half

        next_step, next_batch = next(data_iter)
        timer.epoch = next_step

        with timer("compute_grad"):
            last_loss, next_gradients, state = task.loss_and_gradient(
                parameters, state, next_batch
            )

        with timer("update_momentum"):
            prev_momentum = [z.clone() for z in momentum]
            for z, g in zip(momentum, next_gradients):
                z.mul_(beta).add_(g, alpha=1 - beta)
            momentum_delta = [
                z - z_prev for z, z_prev in zip(momentum, prev_momentum)
            ]

        with timer("communication.tracking"):
            tracking_half = [
                y + dz for y, dz in zip(tracking, momentum_delta)
            ]
            tracking_lower_half = [
                y_l + dz for y_l, dz in zip(tracking_lower, momentum_delta)
            ]
            tracking = lca_update(tracking_gossip, tracking_half, tracking_lower_half)
            tracking_lower = tracking_half

        step = next_step
        batch = next_batch
        gradients = next_gradients

        yield (
            TrainStats(step, bytes_sent()),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )


def gradient_tracking(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )
    correction_gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    correction = [torch.zeros_like(p) for p in parameters]

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent + correction_gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):                            # 更新x_{t+1/2} = x_{t} - partial f(x_t)
            prev_parameters = [p.clone() for p in parameters]
            # for p, c in zip(parameters, gradients):
            #     p.add_(-c,alpha=config["learning_rate"] * learning_rate_schedule(config, step))
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            updates = [p - prev for p, prev in zip(parameters, prev_parameters)]            # partial f(x_t)
            for p, c in zip(parameters, correction):                                        # x_{t} - partial f(x_t) + correlation
                p.add_(c)

        with timer("communication.parameters"):                 # x_{t+1} = Wx_{t+1/2}
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)

        with timer("communication.correction"):
            buffer, shapes = pack([c + u for c, u in zip(correction, updates)])
            correction_gossip.send(buffer)
            correction_gossip.gossip_update(buffer)
            correction = [b - u for b, u in zip(unpack(buffer, shapes), updates)]
            
def d2(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    correction = [torch.zeros_like(p) for p in parameters]

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            prev_parameters = [p.clone() for p in parameters]
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            updates = [p - prev for p, prev in zip(parameters, prev_parameters)]
            for p, c in zip(parameters, correction):
                p.data.add_(c)

        with timer("communication"):
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)

        with timer("update_correction"):
            correction = [
                p - prev - u
                for p, prev, u in zip(unpack(buffer, shapes), prev_parameters, updates)
            ]
            
        # with timer("global_loss_calculation"):
        #     parameters_meanx = [p.clone() for p in parameters]
        #     buffer, shapes = pack(parameters_meanx)
        #     torch.distributed.all_reduce(buffer)
        #     buffer /= get_world_size()
        #     parameters_meanx = unpack(buffer, shapes)
        #     last_loss_meanx, _, _ = task.loss_and_gradient(
        #         parameters_meanx, state, batch
        #     )
            
            
def edm(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    correction = [torch.zeros_like(p) for p in parameters]
    momentum = [torch.zeros_like(p) for p in parameters]

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            for m, g in zip(momentum, gradients):
                m = m.mul_(config["momentum"]).add_(g, alpha = 1-config["momentum"])
            prev_parameters = [p.clone() for p in parameters]
            base_optimizer.step(
                parameters,
                momentum,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            updates = [p - prev for p, prev in zip(parameters, prev_parameters)]
            for p, c in zip(parameters, correction):
                p.data.add_(c)

        with timer("communication"):
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)

        with timer("update_correction"):
            correction = [
                p - prev - u
                for p, prev, u in zip(unpack(buffer, shapes), prev_parameters, updates)
            ]
            

def caedm(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    assert config["base_optimizer"] == "SGD"
    last_loss = None

    topology = configure_topology(config)
    assert not isinstance(topology, list)
    eta_w = config.get("lca_eta_w", config.get("eta_w", None))
    if eta_w is None:
        if topology.num_workers <= 1:
            lambda_w = 0.0
        else:
            w = topology.gossip_matrix(get_gossip_weight(config))
            eigenvalues = np.linalg.eigvalsh(w.detach().cpu().numpy())
            lambda_w = sorted(np.abs(eigenvalues).tolist())[-2]
        eta_w = 1.0 / (1.0 + math.sqrt(max(0.0, 1.0 - lambda_w * lambda_w)))

    if topology.num_workers <= 1:
        gossip = None
    else:
        gossip = MultiTopologyGossipMechanism(
            topology,
            gossip_matrix=get_gossip_weight(config),
            message_drop_prob=config["simulated_dropped_message_probability"],
        )

    def lca_combine(values, lower_values):
        buffer, shapes = pack(values)
        if gossip is not None:
            gossip.send(buffer)
            gossip.gossip_update(buffer)
        lower_buffer, _ = pack(lower_values)
        buffer.mul_(1 + eta_w).add_(lower_buffer, alpha=-eta_w)
        return unpack(buffer, shapes)

    def bytes_sent():
        return 0 if gossip is None else gossip.bytes_sent

    parameters, state = task.initialize(seed=config["seed"])
    lower_parameters = [p.clone() for p in parameters]
    psi = [p.clone() for p in parameters]
    momentum = [torch.zeros_like(p) for p in parameters]

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, bytes_sent()),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            beta = config["momentum"]
            lr = config["learning_rate"] * learning_rate_schedule(config, step)
            for m, g in zip(momentum, gradients):
                m.mul_(beta).add_(g, alpha=1 - beta)
            next_psi = [
                x.clone().add_(m, alpha=-lr)
                for x, m in zip(parameters, momentum)
            ]
            phi = [
                ps_next + x - ps
                for ps_next, x, ps in zip(next_psi, parameters, psi)
            ]
            lower_phi = [
                ps_next + x_lower - ps
                for ps_next, x_lower, ps in zip(next_psi, lower_parameters, psi)
            ]

        with timer("communication"):
            parameters = lca_combine(phi, lower_phi)
            lower_parameters = phi
            psi = [p.clone() for p in next_psi]


def decent_lam(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    aux = [torch.zeros_like(p) for p in parameters]
    momentum = [torch.zeros_like(p) for p in parameters]

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):                            # 更新x_{t+1/2} = x_{t} - partial f(x_t)
            prev_parameters = [p.clone() for p in parameters]
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            
        with timer("communication"):
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)
            
        with timer("final_update"):
            for m, p, p_prev in zip(momentum, parameters, prev_parameters):
                tilde_g = p_prev - p
                m.mul_(config["momentum"]).add_(tilde_g, alpha = 1-config["momentum"])
            parameters = [p.clone() for p in prev_parameters]
            for m,p in zip(momentum, parameters):
                p.sub_(m)
                
        # with timer("global_loss_calculation"):
        #     parameters_meanx = [p.clone() for p in parameters]
        #     buffer, shapes = pack(parameters_meanx)
        #     torch.distributed.all_reduce(buffer)
        #     buffer /= get_world_size()
        #     parameters_meanx = unpack(buffer, shapes)
        #     last_loss_meanx, _, _ = task.loss_and_gradient(
        #         parameters_meanx, state, batch
        #     )

def quasi_global_momentum(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    assert config["base_optimizer"] == "SGD"

    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(
        topology,
        gossip_matrix=get_gossip_weight(config),
        message_drop_prob=config["simulated_dropped_message_probability"],
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        lr = config["learning_rate"] * learning_rate_schedule(config, step)

        with timer("local_update"):
            prev_params = [p.data.clone() for p in parameters]
            prev_optimizer_state = [m.clone() for m in base_optimizer_state]
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            base_optimizer_state = prev_optimizer_state  # restore

        with timer("communication"):  # parameters
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)

        with timer("update_momentum"):
            for m, p, prev in zip(base_optimizer_state, parameters, prev_params):
                m.mul_(config["momentum"]).add_(
                    prev - p, alpha=(1 - config["momentum"]) / max(lr, 1e-8)
                )


def push_sum(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    bytes_sent = 0
    last_loss = None

    assert config["topology"] == "exponential"

    d = int(math.log2(get_world_size()))
    assert 2 ** d == get_world_size()

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)

    for i, (step, batch) in enumerate(
        task.data.iterator(
            batch_size=config["batch_size"],
            shuffle=True,
            ref_num_data=task.mean_num_data_per_worker,
        )
    ):
        timer.epoch = step
        yield (
            TrainStats(step, bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )

        for j in range(config["push_sum_avg_steps"]):
            with timer("communication"):
                send_buffer, shapes = pack(parameters)

                offset = 2 ** (i * (config["push_sum_avg_steps"] + j) % d)
                n = get_world_size()

                # Send
                send_request_handles = []
                neighbor = int(get_rank() + offset) % n
                handle = isend(send_buffer, neighbor)
                bytes_sent += num_bytes(send_buffer)
                send_request_handles.append(handle)

                # Receive
                recv_buffer = torch.empty_like(send_buffer)
                neighbor = int(get_rank() - offset) % n
                recv(recv_buffer, neighbor)
                if random.uniform(0, 1) > config["simulated_dropped_message_probability"]:
                    avg_buffer = send_buffer * 0.5
                    avg_buffer.add_(recv_buffer, alpha=0.5)
                    for handle in send_request_handles:
                        handle.wait()
                    del send_buffer
                else:
                    avg_buffer = send_buffer
                    for handle in send_request_handles:
                        handle.wait()

                parameters = unpack(avg_buffer, shapes)


def learning_rate_schedule(config, epoch):
    """Apply any learning rate schedule"""
    lr = 1.0

    # if config["distributed_world_size"] > 1 and config["num_lr_warmup_epochs"] > 0:
    #     warmup_epochs = config["num_lr_warmup_epochs"]
    #     max_factor = 1.0
    #     factor = 0 + (max_factor - 0) * min(epoch / warmup_epochs, 1.0)
    #     lr *= factor

    for (milestone, factor) in config["lr_schedule_milestones"]:
        if epoch >= milestone:
            lr *= factor
        else:
            return lr
    return lr
