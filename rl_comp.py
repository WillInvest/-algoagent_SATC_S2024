
"""PPO but trying the parellel subprocess function"""
import argparse
import os
import pprint
from threading import Thread
import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, AsyncCollector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor as discrete_Actor, Critic as discrete_Critic
#from shared_policy import shared_env_mm as shared_Env
from time import sleep
import time
import shift
from F_RL_env_mm import SHIFT_env as Env
from lt_rl_env import SHIFT_env as lt_env
from test_flash_crash import flash_crash
import json
import sys

#TODO
#record state action reward next state
#make hidden-size
#research mean field game with shared policy
#mm and lt run
#Sampling the identifiers in a distrbiution so it chagnes while training

def get_args(inputs):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--task', type=str, default='Pendulum-v1')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=inputs["buffer_size"])
    parser.add_argument('--lr', type=float, default=inputs["lr"])
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=inputs["epoch"])
    parser.add_argument('--step-per-epoch', type=int, default=inputs["step_per_epoch"])
    parser.add_argument('--episode-per-collect', type=int, default=inputs["episode_per_collect"])
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=inputs["batch_size"])
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64]) #make larger
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--eps-clip', type=float, default=0.3)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.9)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")

    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(trader, trader_id, identifiers, agent_type, agent_setup, ticker, args):
    test_steps = 36000
    start_time = time.time()
    try:
        #########################Market Maker################################
        if agent_type == "mm":
            #initiate envs
            env = Env(trader = trader, symbol=ticker,
                rl_t = agent_setup["rl_t"],
                info_step= agent_setup["info_step"],
                nTimeStep=agent_setup["nTimeStep"],
                ODBK_range=agent_setup["ODBK_range"],
                max_order_list_length=agent_setup["max_order_list_length"],
                weight = identifiers[0],
                gamma = identifiers[1],#increased from 0.09 to 0.15
                tar_m = identifiers[2],#increased from 0.6
                action_sym = identifiers[3])
            
            args.state_shape = env.observation_space.shape or env.observation_space.n
            args.action_shape = env.action_space.shape or env.action_space.n
            args.max_action = env.action_space.high[0]
            print(f"state_shape{args.state_shape}")
            print(f"action_shape{args.action_shape}")
            train_envs = [env]

            net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
            actor = ActorProb(
                net, args.action_shape, max_action=args.max_action, device=args.device
            ).to(args.device)
            critic = Critic(
                Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
                device=args.device
            ).to(args.device)
            actor_critic = ActorCritic(actor, critic)
            # orthogonal initialization
            for m in actor_critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

            # replace DiagGuassian with Independent(Normal) which is equivalent
            # pass *logits to be consistent with policy.forward
            def dist(*logits):
                #print(*logits)
                #print(Independent(Normal(*logits), 1))
                return Independent(Normal(*logits), 1)
        #########################Liqudity Taker################################
        elif agent_type == "lt":
            # initiate envs
            env = lt_env(trader=trader,  symbol=ticker,
                step_time=agent_setup["step_time"],
                normalizer=agent_setup["normalizer"],
                order_book_range=agent_setup["order_book_range"],

                order_size=identifiers[4],
                target_buy_sell_flows=identifiers[2],
                switch_steps=identifiers[3],
                risk_aversion=identifiers[1], #increased from 0.5
                pnl_weighting=identifiers[0],
                max_iterations=None
            )
            train_envs = [env]
            args.state_shape = env.observation_space.shape or env.observation_space.n
            args.action_shape = env.action_space.shape or env.action_space.n
            print(f"state shape: {args.state_shape}")
            print(f"action shape: {args.action_shape}")
            # model
            net = Net(
                args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device,
                softmax=True, norm_layer=torch.nn.InstanceNorm1d, #norm_args=args.state_shape,
            )
            actor = discrete_Actor(net, args.action_shape, device=args.device).to(args.device)
            critic = discrete_Critic(
                net,
                device=args.device
            ).to(args.device)
            actor_critic = ActorCritic(actor, critic)
            # orthogonal initialization
            for m in actor_critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

            dist = torch.distributions.Categorical
        

        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist, #torch.distributions.Normal, 
            discount_factor=args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            reward_normalization=args.rew_norm,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv,
            dual_clip=args.dual_clip,
            value_clip=args.value_clip,
            gae_lambda=args.gae_lambda,
            action_space=env.action_space,
        )
        train_envs = DummyVectorEnv(train_envs)
        # collector
        train_collector = Collector(#Async
            policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs), exploration_noise=True)#
        )
        #test_collector = Collector(policy, test_envs)

        #for simulation looping purpose:
        # if len(sys.argv) == 2:
        #     curr_run_num = sys.argv[1]
        #     # log
        #     base_path = "/home/shiftpub/Results_Simulation"
        #     log_path = os.path.join(base_path, "log")
        #     if curr_run_num == 1:
        #         last_checkpoint_path = os.path.join(log_path, f"checkpoint/checkpoint")
        #     else:
        #         last_checkpoint_path = os.path.join(log_path, f"checkpoint/checkpoint_{curr_run_num-1}")
        #     current_checkpoint_path = os.path.join(log_path, f"checkpoint/checkpoint_{curr_run_num}")
        #     event_path = os.path.join(log_path, "event")
        #     policy_path = os.path.join(base_path, "Policies")
        #     iteration_info_path = os.path.join(base_path, f"iteration_info/iteration_info_{curr_run_num}")

        #     os.makedirs(current_checkpoint_path)
        #     os.makedirs(iteration_info_path)
                
        # else:
        # log
        base_path = "/home/myteam/shift_comp_rl/Results_Simulation"
        log_path = os.path.join(base_path, "log")
        last_checkpoint_path = os.path.join(log_path, "checkpoint")
        current_checkpoint_path = os.path.join(last_checkpoint_path, "comp_checkpoint")#last_checkpoint_path
        event_path = os.path.join(log_path, "event")
        policy_path = os.path.join(base_path, "Policies")
        iteration_info_path = os.path.join(base_path, "iteration_info")

        # #remove all files in event folder:
        # files = os.listdir(event_path)
        # for file_name in files:
        #     file_path = os.path.join(event_path, file_name)
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)

        writer = SummaryWriter(event_path)
        logger = TensorboardLogger(writer)#, save_interval=args.save_interval

        def save_best_fn(policy):
            print(f"policy saved_{trader_id}")
            torch.save(policy.state_dict(), os.path.join(policy_path, f"best_policy{trader_id}.pth"))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            ckpt_path = os.path.join(current_checkpoint_path, f"checkpoint_{trader_id}.pth")
            print("saved checkpoint")
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim": optim.state_dict(),
                }, ckpt_path
            )
            return ckpt_path

        if args.resume:
            # load from existing checkpoint
            print(f"Loading agent under {last_checkpoint_path}")
            ckpt_path = os.path.join(last_checkpoint_path, f"checkpoint_{trader_id}.pth")
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=args.device)
                policy.load_state_dict(checkpoint["model"])
                optim.load_state_dict(checkpoint["optim"])
                print("Successfully restore policy and optim.***************************")
            else:
                print("Fail to restore policy and optim.********************************")
        #trainer
        trainer = OnpolicyTrainer(
            policy,
            train_collector,
            None,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=5,
            #episode_per_collect=args.episode_per_collect,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=args.resume,
            save_checkpoint_fn=save_checkpoint_fn,
        )

        for epoch, epoch_stat, info in trainer:
            print(f"Epoch: {epoch}")
            print(epoch_stat)
            print(info)

        #assert stop_fn(info["best_reward"])
        torch.save(policy.state_dict(), os.path.join(policy_path, f"policy{trader_id}.pth"))
        
        if __name__ == "__main__":
            #pprint.pprint(info)
            # Let's watch its performance!
            #env = gym.make(args.task)
            policy.eval()
            #collector = Collector(policy, env)
            print("start colleting result")
            # result = train_collector.collect(n_step = test_steps)
            # print(result)
            #print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
            
    except (KeyboardInterrupt, SystemExit):
        print("thread error catched")
        trader.disconnect()
        env.save_to_csv(f"{iteration_info_path}/sep_trader_{trader_id}.csv")
        torch.save(policy.state_dict(), os.path.join(current_checkpoint_path, f"policy_ppo_temp{trader_id}.pth"))
      
    finally:
        end_time = time.time()
        running_time_seconds = end_time - start_time
        running_time_hours = running_time_seconds // 3600
        running_time_minutes = (running_time_seconds % 3600) // 60
        print(f"Total running time: {int(running_time_hours)} hours and {int(running_time_minutes)} minutes.")
        #env.save_to_csv(f"{iteration_info_path}/sep_trader_{trader_id}.csv")

def test_ppo_resume(trader, trader_id, w, agent_type, ticker, args):
    args.resume = True
    test_ppo(trader, trader_id, w, agent_type, ticker, args)


if __name__ == "__main__":
    tickers = [f"CS{sys.argv[1]}"]#,"CS2"
    lt_flows = [[0.3, 0.4, 0.4, 0.4],
                [0.4, 0.35, 0.4, 0.3]] 
    
    #make folder for simulation results
    base_path = "/home/myteam/shift_comp_rl/Results_Simulation"
    log_path = os.path.join(base_path, "log")
    last_checkpoint_path = os.path.join(log_path, "checkpoint")
    current_checkpoint_path = current_checkpoint_path = os.path.join(last_checkpoint_path, "comp_checkpoint")
    event_path = os.path.join(log_path, "event")
    policy_path = os.path.join(base_path, "Policies")
    iteration_info_path = os.path.join(base_path, "iteration_info")
    #check if the base path exist, if not make those directory
    if not os.path.exists(base_path): os.makedirs(base_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    if not os.path.exists(last_checkpoint_path): os.makedirs(last_checkpoint_path)
    if not os.path.exists(current_checkpoint_path): os.makedirs(current_checkpoint_path)
    if not os.path.exists(event_path): os.makedirs(event_path)
    if not os.path.exists(policy_path): os.makedirs(policy_path)
    if not os.path.exists(iteration_info_path): os.makedirs(iteration_info_path)

    config_dic = {
        "agents": ["mm", "mm", "mm", "mm", "lt", "lt", "lt", "lt","lt", "lt", "lt", "lt", "lt","lt"],
        "agent_identifiers": [#MM: weight, gamma,  tar_m,  act_sym_range
                                [0.5,   0.15,   0.5,    [-1,1]],                
                                [0.5,   0.15,   0.5,    [-1,1]],                
                                [0.5,   0.15,   0.5,    [-1,1]],                
                                [0.5,   0.15,   1,      [-1,2]],                
                            #LT: weight, gamma,  targetBuyFrac, frac_duration, order_size
                                [0.5,   0.9,    [[0.2],[0.8]], 0,           3],            
                                [0.5,   0.9,    [[0.4],[0.6]], 0,           3],            
                                [0.5,   0.9,    [[0.5],[0.5]], 0,            3],            
                                [0.5,   0.9,    [[0.6],[0.4]], 0,            3],            
                                [0.5,   0.9,    [[0.8],[0.2]], 0,            3],            
                                [0.5,   0.9,    [[0.2],[0.8]], 0,           3],            
                                [0.5,   0.9,    [[0.4],[0.6]], 0,           3],            
                                [0.5,   0.9,    [[0.5],[0.5]], 0,            3],            
                                [0.5,   0.9,    [[0.6],[0.4]], 0,            3],            
                                [0.5,   0.9,    [[0.8],[0.2]], 0,            3]],
                                # [0,   0.9,    lt_flows, 10000,           18],            
                                # [0,   0.9,    lt_flows, 10000,           18],            
                                # [0,   0.9,    lt_flows, 10000,            18],            
                                # [0,   0.9,    lt_flows, 10000,            18],            
                                # [0,   0.9,    lt_flows, 10000,            18],            
                                # [0,   0.9,    lt_flows, 10000,           18],            
                                # [0,   0.9,    lt_flows, 10000,           18],            
                                # [0,   0.9,    lt_flows, 10000,            18],            
                                # [0,   0.9,    lt_flows, 10000,            18],            
                                # [0,   0.9,    lt_flows, 10000,            18]],
        "mm_unified_setup": {"rl_t": 1, "info_step": 0.25, "nTimeStep": 10, "ODBK_range": 5, "max_order_list_length":30},
        "lt_unified_setup": {"step_time": 1, "normalizer":0.01, "order_book_range":5},
        "resume": True,
        "seed": 0,
        "train_args": { "lr": 0.000,
                        "epoch": 720,
                        "step_per_epoch": 100,
                        "buffer_size": 5,
                        "batch_size": 64,
                        "episode_per_collect": 20},
        "is_flash": False,
        "flash_orders": {"buy/sell": "sell",
                         "flash_size": 1500,
                         "num_orders": 5,
                         "time_bet_order": 1,
                         "num_flash":88,
                         "time_bet_flash": 400},
    }
    with open('/home/myteam/shift_comp_rl/Results_Simulation/iteration_info/config.json', 'w') as json_file:
        json.dump(config_dic, json_file)

    num_proc = len(config_dic["agents"])
    trader_list = []
    threads = []
    #agent identifiers:
             
    identifiers = config_dic["agent_identifiers"]
                #MM:weight, gamma,  tar_m,  act_sym("reg","far")
                #         [0.8,   0.15,   0.5,    [-1,1]],                #1
                #         [0.7,   0.15,   0.5,    [-1,1]],                #2
                #         [0.2,   0.15,   0.5,    [-1,1]],                #3
                #         [0.2,   0.15,   1,      [-1,2]],                #4
                #    #LT:weight, gamma,  targetBuy/SellFlows,  switch_steps, order_size
                #         [0,   0.9,    lt_flows,          12000,        10],            #1
                #         [0,   0.9,    lt_flows,          12000,        10],            #2
                #         [0,   0.9,    lt_flows,          12000,        10],            #3
                #         [0,   0.9,    lt_flows,          12000,         10],            #4
                #         [0,   0.9,    lt_flows,          12000,         10],            #5
                #         [0,   0.9,    lt_flows,          12000,         10],            #6
                #         [0,   0.9,    lt_flows,          12000,         10],            #7
                #         [0,   0.9,    lt_flows,          12000,         10],            #8
                #         [0,   0.9,    lt_flows,          12000,         10],            #9
                #         [0,   0.9,    lt_flows,          12000,         10] ]           #10
                
    agent_type = config_dic["agents"]
    args = get_args(config_dic["train_args"])
    args.resume = config_dic["resume"]  #resume or not
    torch.manual_seed(config_dic["seed"])

    for ticker in tickers:
        if ticker == "CS1": mm_index, lt_index, mm_id, lt_id = 0, 0, 0, 0
        elif ticker == "CS2": mm_index, lt_index, mm_id, lt_id = 4, 10, 4, 10
        for i in range(num_proc):
            if agent_type[i] == "mm":
                num = "{:02d}".format(mm_index+1)
                trader_list.append(shift.Trader(f"marketmaker_rl_{num}"))
                threads.append(Thread(target=test_ppo,args =(trader_list[i], f"mm{mm_id+1}", identifiers[i], agent_type[i], config_dic["mm_unified_setup"], ticker, args)))        
                mm_index += 1
                mm_id += 1

            if agent_type[i] == "lt":
                num = "{:02d}".format(lt_index+1)
                trader_list.append(shift.Trader(f"liquiditytaker_rl_{num}"))
                threads.append(Thread(target=test_ppo,args =(trader_list[i], f"lt{lt_id+1}", identifiers[i], agent_type[i], config_dic["lt_unified_setup"], ticker, args)))        
                lt_index += 1
                lt_id += 1

    try:
        for i in range(len(trader_list)):#len(tickers)*
            trader_list[i].disconnect()
            trader_list[i].connect("/home/myteam/shift_comp_rl/initiator.cfg", "password")
            trader_list[i].sub_all_order_book()
            sleep(1)
            print(f"bp of {i+1}:",trader_list[i].get_portfolio_summary().get_total_bp())
        
        if config_dic["is_flash"]:
            #connect flash crash agent
            crash_maker = shift.Trader("flash_crash_maker_01")
            crash_maker.disconnect()
            crash_maker.connect("/home/myteam/shift_comp_rl/initiator.cfg", "password")
            crash_maker.sub_all_order_book()

            flash_config = config_dic["flash_orders"]
            flash_crash_thread = Thread(target=flash_crash, args=(crash_maker,  flash_config["buy/sell"], flash_config["flash_size"], flash_config["num_orders"], flash_config["time_bet_order"],
                                                                flash_config["num_flash"], flash_config["time_bet_flash"], "CS1"))

        # subscribe order book for all tickers
        # start_time = datetime.now().replace(minute = 15, second = 0)
        # while datetime.now() < start_time:
        #     sleep(1)
        print("Starting")

        for thread in threads:  
            thread.start()
            sleep(0.01)
        if config_dic["is_flash"]:
            flash_crash_thread.start()

        for thread in threads:
            thread.join()
        if config_dic["is_flash"]:
            flash_crash_thread.join()
        
    except KeyboardInterrupt:
        trader_list[0].disconnect()

    finally:
        trader_list[0].disconnect()
