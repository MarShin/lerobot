import yaml
try:
    yaml.safe_load("\x1b")
except Exception as e:
    print(e)

try:
    yaml.safe_load("\x1b--robot.type=so100_follower")
except Exception as e:
    print(e)
