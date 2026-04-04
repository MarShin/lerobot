from robocrew.core.camera import RobotCamera
from robocrew.core.LLMAgent import LLMAgent
from robocrew.robots.XLeRobot.tools import create_move_forward, create_turn_right, create_turn_left
from robocrew.robots.XLeRobot.servo_controls import ServoControler

# 📷 Set up main camera
main_camera = RobotCamera(2)  # camera usb port Eg: /dev/video0

# 🎛️ Set up servo controller
right_arm_wheel_usb = "/dev/tty.usbmodem5B140298121"
left_arm_wheel_usb = "/dev/tty.usbmodem5B140300111"  # provide your right arm usb port. Eg: /dev/ttyACM1
servo_controler = ServoControler(right_arm_wheel_usb=right_arm_wheel_usb, left_arm_head_usb=left_arm_wheel_usb)

# 🛠️ Set up tools
move_forward = create_move_forward(servo_controler)
turn_left = create_turn_left(servo_controler)
turn_right = create_turn_right(servo_controler)

# 🤖 Initialize agent
agent = LLMAgent(
    model="google_genai:gemini-3-flash-preview",
    tools=[move_forward, turn_left, turn_right],
    main_camera=main_camera,
    servo_controler=servo_controler,
)

# 🎯 Give it a task and go!
agent.task = "Raise left arm. And then raise your right arm."
agent.go()