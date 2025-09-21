# SPDX-FileCopyrightText: Copyright (c) 2024 LIMX DYNAMICS
# Configuration for SF_TRON1A robot (bipedal robot with single arm)

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class LimxRoughCfg(LeggedRobotCfg):
    """Configuration for the LIMX SF_TRON1A robot"""
    
    class goal_ee:
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        collision_upper_limits = [0.1, 0.2, -0.05]
        collision_lower_limits = [-0.8, -0.2, -0.7]
        underground_limit = -0.7
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.0  # No pitch offset for bipedal robot
        
        class sphere_center:
            x_offset = 0.3  # Relative to base
            y_offset = 0  # Relative to base
            z_invariant_offset = 0.5  # Relative to terrain
        
        class ranges:
            init_pos_start = [0.4, np.pi/8, 0]
            init_pos_end = [0.6, 0, 0]
            pos_l = [0.3, 0.8]
            pos_p = [-1 * np.pi / 2.5, 1 * np.pi / 3]
            pos_y = [-1.0, 1.0]
            
            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            final_tracking_ee_reward = 0.55
        
        sphere_error_scale = [1, 1, 1]
        orn_error_scale = [1, 1, 1]
    
    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
    
    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3.  # time before command are changed[s]
        
        lin_vel_x_schedule = [0, 0.5]
        ang_vel_yaw_schedule = [0, 1]
        tracking_ang_vel_yaw_schedule = [0, 1]
        
        ang_vel_yaw_clip = 0.5
        lin_vel_x_clip = 0.2
        
        class ranges:
            lin_vel_x = [-0.5, 0.5]  # Slower for bipedal robot
            ang_vel_yaw = [-0.8, 0.8]
    
    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
    
    class env:
        num_envs = 4096
        num_actions = 6 + 8  # 6 arm joints + 8 leg joints (与代码一致)
        num_torques = 6 + 8
        action_delay = -1  # 早期去除动作延迟，便于学习稳定
        num_gripper_joints = 0  # No gripper in this model
        num_proprio = 2 + 3 + 14 + 14 + 8 + 2 + 3 + 3 + 3  # Updated for bipedal robot
        num_priv = 5 + 1 + 8  # mass_params(5) + friction(1) + leg_motor_strength(8)
        history_len = 10
        num_observations = num_proprio * (history_len+1) + num_priv
        num_privileged_obs = None
        send_timeouts = True
        episode_length_s = 10
        reorder_dofs = True
        teleop_mode = False
        record_video = False
        stand_by = False
        observe_gait_commands = True  # 启用步态相位与接触塑形
        frequencies = 2
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.6]  # Higher initial position for bipedal robot
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # Left leg
            'abad_L_Joint': 0.0,
            'hip_L_Joint': -0.2,
            'knee_L_Joint': 0.4,
            'ankle_L_Joint': -0.2,
            
            # Right leg
            'abad_R_Joint': 0.0,
            'hip_R_Joint': -0.2,
            'knee_R_Joint': 0.4,
            'ankle_R_Joint': -0.2,
            
            # Arm joints
            'J1': 0.0,
            'J2': -1.0,
            'J3': 1.5,
            'J4': 0.0,
            'J5': 0.0,
            'J6': 0.0,
        }
        rand_yaw_range = np.pi/4
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1
    
    class control:
        stiffness = {'leg': 120, 'arm': 10}
        damping = {'leg': 3.5, 'arm': 1.0}
        
        adaptive_arm_gains = False
        # action scale: target angle = actionScale * action + defaultAngle
        # First 6 are arm, last 8 are legs
        action_scale = [1.5, 0.8, 0.8, 0.5, 0.5, 0.5] + [0.5, 0.5, 0.5, 0.3] * 2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_supervision = False
    
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/SF_TRON1A/urdf/robot.urdf'
        foot_name = "ankle"
        gripper_name = "link6"  # End effector link
        penalize_contacts_on = ["hip", "knee", "abad"]
        terminate_after_contacts_on = ["base_Link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        collapse_fixed_joints = True
        fix_base_link = False
    
    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16
    
    class arm:
        init_target_ee_base = [0.3, 0.0, 0.2]
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)
    
    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0]
        randomize_base_mass = True
        added_mass_range = [0., 10.]  # Less additional mass for bipedal
        randomize_base_com = True
        added_com_range_x = [-0.1, 0.1]
        added_com_range_y = [-0.1, 0.1]
        added_com_range_z = [-0.1, 0.1]
        randomize_motor = False  # 先关闭电机随机化，稳定早期学习
        leg_motor_strength_range = [0.9, 1.1]
        arm_motor_strength_range = [0.9, 1.1]
        randomize_gripper_mass = False  # No gripper
        gripper_added_mass_range = [0, 0]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.3  # Less push for bipedal stability
    
    class rewards:
        reward_container_name = "limx_rewards"
        
        # Common parameters
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.4
        base_height_target = 0.45  # Lower for bipedal
        max_contact_force = 40.
        
        # Gait control parameters (for bipedal walking)
        gait_vel_sigma = 0.5
        gait_force_sigma = 0.5
        kappa_gait_probs = 0.07
        feet_height_target = 0.15  # Lower for bipedal
        
        feet_aritime_allfeet = False
        feet_height_allfeet = False
        
        class scales:
            # Gait control rewards
            tracking_contacts_shaped_force = 2.0  # 奖励函数返回为负，系数取正以形成惩罚
            tracking_contacts_shaped_vel = 2.0
            feet_air_time = 2.0
            feet_height = 1.0
            
            # Tracking rewards
            tracking_lin_vel_max = 2.0
            tracking_lin_vel_x_l1 = 0.
            tracking_lin_vel_x_exp = 0
            tracking_ang_vel = 0.5
            
            # Stability rewards (more important for bipedal)
            orientation = -2.0
            base_height = -2.0
            roll = -1.5
            pitch = -1.5
            
            # Common rewards
            delta_torques = -1.0e-7/4.0
            work = -0.003
            energy_square = 0.0
            torques = -2.5e-5
            stand_still = 1.0
            walking_dof = 1.5
            dof_default_pos = 0.0
            dof_error = 0.0
            alive = 1.0
            lin_vel_z = -2.0  # Higher penalty for vertical velocity
            
            # Action and motion smoothness
            ang_vel_xy = -0.3
            dof_acc = -7.5e-7
            collision = -10.
            action_rate = -0.015
            dof_pos_limits = -5.0
            hip_pos = -0.3
            feet_jerk = -0.0002
            feet_drag = -0.08
            feet_contact_forces = -0.001
            
            # Walking specific rewards
            orientation_walking = -1.0
            orientation_standing = -1.0
            base_height_walking = -3.0
            base_height_standing = -3.0
            torques_walking = 0.0
            torques_standing = 0.0
            energy_square_walking = 0.0
            energy_square_standing = 0.0
            penalty_lin_vel_y = -0.5
        
        class arm_scales:
            arm_termination = None
            tracking_ee_sphere = 0.
            tracking_ee_world = 0.8
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_orientation = None
            arm_energy_abs_sum = None
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = None
    
    class viewer:
        pos = [-20, 0, 20]  # [m]
        lookat = [0, 0, -2]  # [m]
    
    class termination:
        r_threshold = 0.5  # Tighter for bipedal stability
        p_threshold = 0.5
        z_threshold = 0.2
    
    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1  # for fast
        horizontal_scale = 0.05  # [m] influence computation time by a lot
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        height = [0.00, 0.08]  # Lower terrain for bipedal
        gap_size = [0.02, 0.08]
        stepping_stone_distance = [0.02, 0.06]
        downsampled_scale = 0.075
        curriculum = False
        
        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        measure_heights = True
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())
        slope_treshold = None
        origin_zero_z = False


class LimxRoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        continue_from_last_std = True
        init_std = [[1.0] * 6 + [1.0, 1.0, 1.0, 0.5] * 2]  # 6 arm joints + 8 leg joints
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu'
        output_tanh = False
        
        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]
        
        priv_encoder_dims = [64, 20]
        
        num_arm_actions = 6
        num_leg_actions = 8
        
        adaptive_arm_gains = LimxRoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0
    
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        schedule = 'fixed'
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.2] * 3 + [0.05] * 3 + [0.2, 0.2, 0.2, 0.1] * 2]  # 6 arm + 8 leg
        
        mixing_schedule = [1.0, 0, 3000]
        torque_supervision = LimxRoughCfg.control.torque_supervision
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = LimxRoughCfg.control.adaptive_arm_gains
        
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000]
    
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 45000
        
        # logging
        save_interval = 200
        experiment_name = 'limx_sf_tron1a'
        run_name = ''
        
        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
