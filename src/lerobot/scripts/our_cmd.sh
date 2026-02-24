
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
  --dataset.num_episodes=10 \
  --policy.path=lerobot/smolvla_base \
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
  --dataset.repo_id=marshin68/raise_arms \
  --dataset.num_episodes=25 \
  --dataset.single_task="Raise your arms"


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