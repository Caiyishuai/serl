#!/usr/bin/env python3

import os
import sys
import types

# Fix for JAX CUDA detection issue - must be set before any JAX imports
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

try:
    from nvidia import cuda_nvcc
    # If module exists but __file__ is None, set it to a dummy value
    if not hasattr(cuda_nvcc, '__file__') or cuda_nvcc.__file__ is None:
        cuda_nvcc.__file__ = '/tmp/cuda_nvcc_workaround.py'
except ImportError:
    # Create nvidia package and cuda_nvcc module if they don't exist
    if 'nvidia' not in sys.modules:
        nvidia = types.ModuleType('nvidia')
        sys.modules['nvidia'] = nvidia
    else:
        nvidia = sys.modules['nvidia']
    
    cuda_nvcc = types.ModuleType('cuda_nvcc')
    cuda_nvcc.__file__ = '/tmp/cuda_nvcc_workaround.py'
    nvidia.cuda_nvcc = cuda_nvcc
    sys.modules['nvidia.cuda_nvcc'] = cuda_nvcc

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from absl import app, flags
from flax.training import checkpoints
import pickle as pkl
from tqdm import tqdm

import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.launcher import make_drq_agent
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

import franka_sim

FLAGS = flags.FLAGS

flags.DEFINE_string("replay_env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("collect_env", "PandaPickCubeVisionWithForce-v0", "Name of environment.")



flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory.")
flags.DEFINE_string("reward_type", "dense", "Reward type: 'dense' or 'binary'.")
flags.DEFINE_boolean("save_trajs", True, "Whether to save trajectories.")
flags.DEFINE_boolean("save_success_only", False, "Whether to save only successful trajectories.")
flags.DEFINE_integer("num_episodes", 150, "Number of episodes to evaluate.")
flags.DEFINE_string("output_path", os.path.join(os.path.dirname(__file__), "success_trajs_{}_force_reward.pkl"), "Path to save the collected data. Use {} as placeholder for num_episodes.")
flags.DEFINE_string("fail_output_path", os.path.join(os.path.dirname(__file__), "fail_trajs_{}_force_reward.pkl"), "Path to save the failed trajectories. Use {} as placeholder for num_episodes.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_boolean("render", False, "Whether to render the environment.")
flags.DEFINE_boolean("save_video", True, "Whether to save the evaluation video.")
flags.DEFINE_string("video_path", os.path.join(os.path.dirname(__file__), "eval_video.mp4"), "Path to save the video.")
flags.DEFINE_string("success_video_path", os.path.join(os.path.dirname(__file__), "eval_success_video.mp4"), "Path to save the success video.")
flags.DEFINE_string("fail_video_path", os.path.join(os.path.dirname(__file__), "eval_fail_video.mp4"), "Path to save the fail video.")
flags.DEFINE_boolean("pause_on_success", True, "Whether to pause for 0.5s on success.")
flags.DEFINE_integer("target_success_count", 20, "Target number of successful trajectories to collect.")
flags.DEFINE_integer("target_fail_count", 20, "Target number of failed trajectories to collect.")

def augment_obs_with_force_torque(obs, env):
    """
    将原始环境的observation增强为包含force和torque的格式。
    匹配参考文件格式：state是22维展平数组
    
    State格式 (22维):
    - [0]: gripper_pose (1)
    - [1:4]: tcp_force (3)
    - [4:10]: tcp_pose (6)
    - [10:13]: tcp_torque (3)
    - [13:19]: tcp_vel (6)
    - [19:22]: block_pos (3) - 物体位置信息
    
    Args:
        obs: 原始环境的observation (PandaPickCubeVision-v0格式)
        env: 环境实例
    
    Returns:
        augmented_obs: 增强后的observation，state为展平数组
        original_state_dict: 原始的字典格式state（用于info）
    """
    # 创建增强后的observation副本
    augmented_obs = {}
    original_state_dict = {}
    
    # 复制images（如果存在）
    if "images" in obs:
        augmented_obs["images"] = obs["images"].copy()
    
    # 从Mujoco传感器直接读取所有必要的数据
    try:
        # Gripper pose (1D)
        if "state" in obs and "panda/gripper_pos" in obs["state"]:
            gripper_pose = obs["state"]["panda/gripper_pos"]
        else:
            gripper_pose = np.array([env.unwrapped.data.joint("right_driver_joint").qpos[0]], dtype=np.float32)
        
        if len(gripper_pose.shape) == 0:
            gripper_pose = np.array([gripper_pose], dtype=np.float32)
        
        # TCP force (3D)
        tcp_force = env.unwrapped.data.sensor("panda/wrist_force").data.astype(np.float32)
        
        # TCP pose (position + orientation, 6D)
        tcp_pos = env.unwrapped.data.sensor("2f85/pinch_pos").data.astype(np.float32)
        tcp_quat = env.unwrapped.data.sensor("2f85/pinch_quat").data.astype(np.float32)
        tcp_pose = np.concatenate([tcp_pos, tcp_quat[1:]])  # x,y,z, qx,qy,qz (6D)
        
        # TCP torque (3D)
        tcp_torque = env.unwrapped.data.sensor("panda/wrist_torque").data.astype(np.float32)
        
        # TCP velocity (6D: linear + angular)
        tcp_vel = env.unwrapped.data.sensor("2f85/pinch_vel").data.astype(np.float32)
        tcp_vel_full = np.concatenate([tcp_vel, np.zeros(3, dtype=np.float32)])  # 6D
        
        # Block position (3D) - 物体位置信息
        block_pos = env.unwrapped.data.sensor("block_pos").data.astype(np.float32)
        
        # 保存原始字典格式（用于info）
        original_state_dict = {
            'tcp_pose': tcp_pose,
            'tcp_vel': tcp_vel_full,
            'gripper_pose': gripper_pose,
            'tcp_force': tcp_force,
            'tcp_torque': tcp_torque,
            'block_pos': block_pos,
        }
        
        # 按照格式展平为22维数组
        # [0]: gripper_pose, [1:4]: tcp_force, [4:10]: tcp_pose, [10:13]: tcp_torque, [13:19]: tcp_vel, [19:22]: block_pos
        flattened_state = np.concatenate([
            gripper_pose.flatten(),      # [0] - 1 dim
            tcp_force,                    # [1:4] - 3 dims
            tcp_pose,                     # [4:10] - 6 dims
            tcp_torque,                   # [10:13] - 3 dims
            tcp_vel_full,                 # [13:19] - 6 dims
            block_pos,                    # [19:22] - 3 dims
        ]).astype(np.float32)
        
        augmented_obs["state"] = flattened_state
        
    except Exception as e:
        print(f"Warning: Failed to augment observation with force/torque: {e}")
        import traceback
        traceback.print_exc()
        # 如果出错，返回原始observation和空字典
        return obs, {}
    
    return augmented_obs, original_state_dict

def main(_):
    FLAGS.reward_type = "dense" # "dense" or "01"
    FLAGS.checkpoint_path = "examples/async_drq_sim/checkpoints_saved_80cm_max40cm/checkpoint_396000"
    # FLAGS.checkpoint_path = "examples/async_drq_sim/checkpoints/checkpoint_118000"
    # FLAGS.checkpoint_path="random_checkpoint"

    FLAGS.output_path = os.path.join(os.path.dirname(__file__), f"success_trajs_{FLAGS.target_success_count}_force_{FLAGS.reward_type}.pkl")
    FLAGS.fail_output_path = os.path.join(os.path.dirname(__file__), f"fail_trajs_{FLAGS.target_fail_count}_force_{FLAGS.reward_type}.pkl")


    # Use GPU if available
    if jax.default_backend() == "cpu":
        print("Warning: JAX is using CPU. Inference might be slow.")

    # 验证环境名称格式
    if not FLAGS.replay_env.endswith("-v0"):
        print(f"Warning: replay_env '{FLAGS.replay_env}' should end with '-v0'. Attempting to fix...")
        FLAGS.replay_env = FLAGS.replay_env + "-v0"
        print(f"Fixed to: {FLAGS.replay_env}")
    
    if not FLAGS.collect_env.endswith("-v0"):
        print(f"Warning: collect_env '{FLAGS.collect_env}' should end with '-v0'. Attempting to fix...")
        FLAGS.collect_env = FLAGS.collect_env + "-v0"
        print(f"Fixed to: {FLAGS.collect_env}")

    # Create Environment for replay (model inference)
    # Note: We pass reward_type via kwargs which gym.make passes to the env constructor
    print(f"Creating replay environment {FLAGS.replay_env} for model inference")
    print(f"Data will be saved in {FLAGS.collect_env} format with force/torque")
    
    # Ensure environment is registered
    try:
        env_spec = gym.spec(FLAGS.replay_env)
    except gym.error.NameNotFound:
        print(f"Environment {FLAGS.replay_env} not found. Attempting explicit registration...")
        from gymnasium.envs.registration import register
        
        if FLAGS.replay_env == "PandaPickCubeVision-v0":
            register(
                id="PandaPickCubeVision-v0",
                entry_point="franka_sim.envs.panda_pick_gym_env:PandaPickCubeGymEnv",
                max_episode_steps=200,
                kwargs={"image_obs": True},
            )
            print("Registered PandaPickCubeVision-v0")
        elif FLAGS.replay_env == "PandaPickCubeVisionWithForce-v0":
            register(
                id="PandaPickCubeVisionWithForce-v0",
                entry_point="franka_sim.envs.panda_pick_gym_env_with_force:PandaPickCubeGymEnvWithForce",
                max_episode_steps=200,
                kwargs={"image_obs": True},
            )
            print("Registered PandaPickCubeVisionWithForce-v0")
    
    # Use replay_env (original env) for running the model
    env = gym.make(FLAGS.replay_env, render_mode="rgb_array", reward_type=FLAGS.reward_type) 
    
    # Wrappers
    if FLAGS.replay_env in ["PandaPickCubeVision-v0", "PandaPickCubeVisionWithForce-v0"]:
        env = SERLObsWrapper(env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    
    # Initialize Agent (needed to restore checkpoint into)
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, key = jax.random.split(rng)
    
    print("Initializing agent...")
    agent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # Restore Checkpoint
    ckpt_path = os.path.abspath(FLAGS.checkpoint_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}. Please check the path.")
    
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # We restore the 'state' part of the agent
    # If ckpt_path is a file, verify it exists. If directory, restore_checkpoint handles it.
    restored_state = checkpoints.restore_checkpoint(ckpt_path, target=agent.state)
    agent = agent.replace(state=restored_state)
    
    # Evaluation Loop
    collected_trajs = []
    collected_fail_trajs = []
    success_count = 0
    fail_count = 0
    success_video_writer = None
    fail_video_writer = None
    can_render = True
    
    # ==================== Phase 1: 使用模型收集成功数据 ====================
    print(f"\n=== Phase 1: 使用模型收集成功数据，同时记录失败数据 ===")
    print(f"目标成功数量: {FLAGS.target_success_count}")
    
    pbar_success = tqdm(total=FLAGS.target_success_count, desc="Success (Phase 1)", position=0)
    pbar_fail = tqdm(total=FLAGS.target_fail_count, desc="Fail (Phase 1)", position=1)
    episode_idx = 0
    
    # Phase 1: 使用模型收集直到成功数量达标
    max_phase1_episodes = FLAGS.target_success_count * 5  # 设置一个上限防止无限循环
    while success_count < FLAGS.target_success_count and episode_idx < max_phase1_episodes:
        episode_idx += 1
        obs, _ = env.reset()
        
        # 获取物体的初始位置信息 (从底层 Mujoco 环境获取)
        # 注意：env 是经过包装的，使用 .unwrapped 访问原始环境
        try:
            init_obj_pos = env.unwrapped.data.sensor("block_pos").data.copy()
        except Exception as e:
            init_obj_pos = None
            print(f"Warning: Could not get initial object position: {e}")

        done = False
        truncated = False
        
        # Trajectory storage
        traj = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'infos': [],
            # 'initial_object_pos': init_obj_pos,  # 单独存放初始位置信息
            # 'block_positions': [],  # 每一步的物体位置信息
            # 'tcp_forces': [],  # 每一步的末端力信息
            # 'tcp_torques': []  # 每一步的末端力矩信息
        }
        
        episode_success = False
        success_step = 0 # number of steps to success
        episode_frames = []  # 存储该episode的所有帧
        
        while not (done or truncated):
            rng, key = jax.random.split(rng)
            
            # 使用模型预测动作
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                argmax=True
            )
            action = np.asarray(jax.device_get(actions))
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            success_step += 1
            
            # 将observation转换为WithForce格式（包含force和torque）
            augmented_obs, original_state_obs = augment_obs_with_force_torque(obs, env)
            augmented_next_obs, original_next_state_obs = augment_obs_with_force_torque(next_obs, env)
            
            # 将原始的字典格式state信息保存到info中
            if original_state_obs:
                info['original_state_obs'] = original_state_obs
            
            # Record transition (保存增强后的observations)
            traj['observations'].append(augmented_obs)
            traj['actions'].append(action)
            traj['rewards'].append(reward)
            traj['next_observations'].append(augmented_next_obs)
            traj['dones'].append(done or truncated)
            traj['infos'].append(info)
            
            if done:
                episode_success = True
            
            obs = next_obs
            
            # Visualization and Video Saving
            img_bgr = None
            if FLAGS.render or FLAGS.save_video:
                if "images" in obs and "front" in obs["images"]:
                    img = obs["images"]["front"]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = env.render()
                    if isinstance(img, list): img = img[0]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if FLAGS.render and img_bgr is not None:
                if not can_render:
                    # 尝试打开窗口来检查显示器是否可用
                    try:
                        cv2.imshow("Evaluation", img_bgr)
                        cv2.waitKey(1)
                        can_render = True
                    except Exception as e:
                        print(f"警告: 无法渲染 - 没有可用的显示器。禁用渲染功能。错误: {e}")
                        can_render = False
                        FLAGS.render = False  # 禁用后续渲染尝试
                else:
                    cv2.imshow("Evaluation", img_bgr)
                    cv2.waitKey(1)
            
            if FLAGS.save_video and img_bgr is not None:
                episode_frames.append(img_bgr.copy())

            if FLAGS.pause_on_success and (done or truncated) and FLAGS.render:
                cv2.waitKey(500) # 500ms = 0.5s

        # Episode结束后，根据成功/失败写入对应的视频
        if FLAGS.save_video and episode_frames:
            if episode_success:
                if success_video_writer is None:
                    height, width, layers = episode_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    success_video_writer = cv2.VideoWriter(FLAGS.success_video_path, fourcc, 20.0, (width, height))
                
                for frame in episode_frames:
                    success_video_writer.write(frame)
                
                # Pause at the end
                if FLAGS.pause_on_success:
                    for _ in range(10):  # 0.5s at 20fps
                        success_video_writer.write(episode_frames[-1])
            else:
                if fail_video_writer is None:
                    height, width, layers = episode_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fail_video_writer = cv2.VideoWriter(FLAGS.fail_video_path, fourcc, 20.0, (width, height))
                
                for frame in episode_frames:
                    fail_video_writer.write(frame)
                
                # Pause at the end
                if FLAGS.pause_on_success:
                    for _ in range(10):
                        fail_video_writer.write(episode_frames[-1])

        if episode_success:
            success_count += 1
            print(f"Success step: {success_step}")
            pbar_success.update(1)
            if FLAGS.save_trajs:
                collected_trajs.append(traj)
        else:
            fail_count += 1
            print(f"Fail step: {success_step}")
            pbar_fail.update(1)
            if FLAGS.save_trajs:
                collected_fail_trajs.append(traj)
    
    pbar_success.close()
    pbar_fail.close()
    
    print(f"\nPhase 1 完成:")
    print(f"成功数量: {success_count}/{FLAGS.target_success_count}")
    print(f"失败数量: {fail_count}/{FLAGS.target_fail_count}")
    
    # ==================== Phase 2: 如果失败数据不足，使用随机动作收集 ====================
    if fail_count < FLAGS.target_fail_count:
        print(f"\n=== Phase 2: 失败数据不足，使用随机动作收集失败数据 ===")
        print(f"当前失败数量: {fail_count}, 目标: {FLAGS.target_fail_count}")
        
        pbar_fail_random = tqdm(total=FLAGS.target_fail_count - fail_count, desc="Fail (Random)", position=0)
        
        max_phase2_episodes = (FLAGS.target_fail_count - fail_count) * 3  # 设置上限
        phase2_episode_idx = 0
        
        while fail_count < FLAGS.target_fail_count and phase2_episode_idx < max_phase2_episodes:
            phase2_episode_idx += 1
            obs, _ = env.reset()
            
            # 获取物体的初始位置信息
            try:
                init_obj_pos = env.unwrapped.data.sensor("block_pos").data.copy()
            except Exception as e:
                init_obj_pos = None
            
            done = False
            truncated = False
            
            # Trajectory storage
            traj = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'dones': [],
                'infos': [],
            }
            
            episode_success = False
            success_step = 0
            episode_frames = []
            
            while not (done or truncated):
                rng, key = jax.random.split(rng)
                
                # 使用随机动作
                action = env.action_space.sample()
                
                # Step environment
                next_obs, reward, done, truncated, info = env.step(action)
                success_step += 1
                
                # 将observation转换为WithForce格式（包含force和torque）
                augmented_obs, original_state_obs = augment_obs_with_force_torque(obs, env)
                augmented_next_obs, original_next_state_obs = augment_obs_with_force_torque(next_obs, env)
                
                # 将原始的字典格式state信息保存到info中
                if original_state_obs:
                    info['original_state_obs'] = original_state_obs
                
                # Record transition (保存增强后的observations)
                traj['observations'].append(augmented_obs)
                traj['actions'].append(action)
                traj['rewards'].append(reward)
                traj['next_observations'].append(augmented_next_obs)
                traj['dones'].append(done or truncated)
                traj['infos'].append(info)
                
                if done:
                    episode_success = True
                
                obs = next_obs
                
                # Visualization and Video Saving
                img_bgr = None
                if FLAGS.render or FLAGS.save_video:
                    if "images" in obs and "front" in obs["images"]:
                        img = obs["images"]["front"]
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    else:
                        img = env.render()
                        if isinstance(img, list): img = img[0]
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                if FLAGS.render and img_bgr is not None:
                    if can_render:
                        cv2.imshow("Evaluation", img_bgr)
                        cv2.waitKey(1)
                
                if FLAGS.save_video and img_bgr is not None:
                    episode_frames.append(img_bgr.copy())
            
            # Episode结束后处理
            if FLAGS.save_video and episode_frames:
                # 只保存失败的轨迹到失败视频
                if not episode_success:
                    if fail_video_writer is None:
                        height, width, layers = episode_frames[0].shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fail_video_writer = cv2.VideoWriter(FLAGS.fail_video_path, fourcc, 20.0, (width, height))
                    
                    for frame in episode_frames:
                        fail_video_writer.write(frame)
                    
                    if FLAGS.pause_on_success:
                        for _ in range(10):
                            fail_video_writer.write(episode_frames[-1])
                # 如果意外成功了，也保存到成功视频
                else:
                    if success_video_writer is None:
                        height, width, layers = episode_frames[0].shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        success_video_writer = cv2.VideoWriter(FLAGS.success_video_path, fourcc, 20.0, (width, height))
                    
                    for frame in episode_frames:
                        success_video_writer.write(frame)
                    
                    if FLAGS.pause_on_success:
                        for _ in range(10):
                            success_video_writer.write(episode_frames[-1])
            
            if episode_success:
                success_count += 1
                print(f"Random action 意外成功! Step: {success_step}")
                # 不保存到轨迹，因为我们已经有足够的成功轨迹
            else:
                fail_count += 1
                print(f"Random fail step: {success_step}")
                pbar_fail_random.update(1)
                if FLAGS.save_trajs:
                    collected_fail_trajs.append(traj)
        
        pbar_fail_random.close()
    
    print(f"\n总体收集完成.")
    total_episodes = episode_idx + (phase2_episode_idx if fail_count < FLAGS.target_fail_count else 0)
    print(f"成功率: {success_count}/{total_episodes} ({success_count/total_episodes*100:.2f}%)")
    print(f"失败数量: {fail_count}/{total_episodes} ({fail_count/total_episodes*100:.2f}%)")
    
    # 将轨迹数据转换为参考文件格式
    def convert_trajs_to_reference_format(trajs):
        """
        将轨迹列表转换为参考文件格式
        
        参考格式：
        - observations: dict with keys ['state', 'images'...], 每个key对应一个数组
        - next_observations: 同上
        - actions, rewards, dones, infos: 数组
        """
        converted_trajs = []
        for traj in trajs:
            # 提取所有observations中的state和images
            states = []
            images_dict = {}
            
            for obs in traj['observations']:
                if 'state' in obs:
                    states.append(obs['state'])
                if 'images' in obs:
                    for img_key, img_val in obs['images'].items():
                        if img_key not in images_dict:
                            images_dict[img_key] = []
                        images_dict[img_key].append(img_val)
            
            # 提取所有next_observations中的state和images
            next_states = []
            next_images_dict = {}
            
            for obs in traj['next_observations']:
                if 'state' in obs:
                    next_states.append(obs['state'])
                if 'images' in obs:
                    for img_key, img_val in obs['images'].items():
                        if img_key not in next_images_dict:
                            next_images_dict[img_key] = []
                        next_images_dict[img_key].append(img_val)
            
            # 构建新的轨迹格式
            converted_traj = {
                'observations': {
                    'state': np.array(states, dtype=np.float32)
                },
                'next_observations': {
                    'state': np.array(next_states, dtype=np.float32)
                },
                'actions': np.array(traj['actions'], dtype=np.float32),
                'rewards': np.array(traj['rewards'], dtype=np.float32),
                'dones': np.array(traj['dones'], dtype=bool),
                'infos': traj['infos']
            }
            
            # 添加images（如果有）
            for img_key, img_list in images_dict.items():
                converted_traj['observations'][img_key] = np.array(img_list, dtype=np.uint8)
            
            for img_key, img_list in next_images_dict.items():
                converted_traj['next_observations'][img_key] = np.array(img_list, dtype=np.uint8)
            
            converted_trajs.append(converted_traj)
        
        return converted_trajs
    
    if FLAGS.save_trajs and collected_trajs:
        # 转换格式
        converted_success_trajs = convert_trajs_to_reference_format(collected_trajs)
        
        # Format the output path with the number of collected successful trajectories
        output_path = FLAGS.output_path.format(success_count)
        with open(output_path, 'wb') as f:
            pkl.dump(converted_success_trajs, f)
        print(f"保存了 {len(converted_success_trajs)} 个成功轨迹到 {output_path}")
    elif FLAGS.save_trajs:
        print("没有成功轨迹被保存.")
    
    if FLAGS.save_trajs and collected_fail_trajs:
        # 转换格式
        converted_fail_trajs = convert_trajs_to_reference_format(collected_fail_trajs)
        
        # Format the output path with the number of collected failed trajectories
        fail_output_path = FLAGS.fail_output_path.format(fail_count)
        with open(fail_output_path, 'wb') as f:
            pkl.dump(converted_fail_trajs, f)
        print(f"保存了 {len(converted_fail_trajs)} 个失败轨迹到 {fail_output_path}")
    elif FLAGS.save_trajs:
        print("没有失败轨迹被保存.")

        
    if success_video_writer is not None:
        success_video_writer.release()
        print(f"成功视频保存到 {FLAGS.success_video_path}")
    
    if fail_video_writer is not None:
        fail_video_writer.release()
        print(f"失败视频保存到 {FLAGS.fail_video_path}")
        
    env.close()
    if FLAGS.render:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(main)

