
  lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5B140298121 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5B140300111 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}
  }' \
  --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
  }' \
   --dataset.rename_map='{"observation.images.left_wrist": "observation.images.camera1", "observation.images.right_wrist": "observation.images.camera2"}' \
  --dataset.single_task="Raise your arms" \
  --dataset.num_episodes=5 \
  --policy.path=marshin68/smolvla_base \
  --dataset.repo_id=marshin68/eval_results 



  lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5B140298121 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5B140300111 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}
  }' \
  --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=keyboard \
  --teleop.id=bimanual_leader \
  --display_data=true \
  --dataset.repo_id=marshin68/bi_so_follower_pick_box2 \
  --dataset.num_episodes=5 \
  --dataset.single_task="pick up the box and put in the grey tray" \
  --dataset.push_to_hub=false


  lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=marshin68/smolvla_base \
  --dataset.repo_id=marshin68/raise_arms \
  --batch_size=2 \
  --steps=2 \
  --output_dir=outputs/train/smolvla_raise_arm \
  --job_name=smolvla_training_raise_arm \
  --policy.device=cuda \
  --rename_map='{"observation.images.left_wrist": "observation.images.camera1", "observation.images.right_wrist": "observation.images.camera2"}' \
  --wandb.enable=true


  python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8999 \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port=/dev/tty.usbmodem5B140298121 \
    --robot.right_arm_config.port=/dev/tty.usbmodem5B140300111 \
    --robot.id=bimanual_follower \
    --robot.left_arm_config.cameras='{
      wrist: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}
    }' \
    --robot.right_arm_config.cameras='{
      wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
    }' \
    --task="open your grippers" \
    --policy_type=smolvla \
    --pretrained_name_or_path=marshin68/smolvla_base \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True