# Semaphore

### A full-body keyboard

[![demo](demo.gif)](https://youtu.be/h376W93gQq4)

View a fuller demo and more background on the project at https://youtu.be/h376W93gQq4

The next iteration of this project, designed as a full-body *game* controller, is also available at https://github.com/everythingishacked/Gamebody

Semaphore uses [OpenCV](https://github.com/opencv/opencv-python) and MediaPipe's [Pose detection](https://google.github.io/mediapipe/solutions/pose.html#python-solution-api) to perform real-time detection of body landmarks from video input. From there, relative differences are calculated to determine specific positions and translate those into keys and commands sent via [keyboard](https://github.com/boppreh/keyboard).

The primary input is to "type" letters, digits, and symbols via [flag semaphore](https://en.wikipedia.org/wiki/Flag_semaphore) by extending both arms at various angles. Rather than waiting a set time after every signal, you can jump to repeat the last sent symbol.

See the `SEMAPHORES` dictionary in the code for a full set
of angles, which mostly conform to standard US semaphore with some custom additions. Most of the rest of the keyboard is included as other modifier gestures, such as:

- `shift`: open both hands, instead of fists
- `backspace`: both hands over mouth
- `digits` and other extra symbols: squat while signaling
- `command`: lift left leg to ~horizontal thigh
- `control`: lift right leg to ~horizontal thigh
- `arrow left/right/up/down`: cross arms and raise each straight leg `LEG_ARROW_ANGLE` degrees
- repeat previous letter/command: jump

Running on latest MacOS from Terminal, toggle the following for keyboard access:
System Settings -> Privacy & Security -> Accessibility -> Terminal -> slide to allow

For Mac, this uses a [custom keyboard library](https://github.com/everythingishacked/keyboard). This is built for a Mac keyboard, but you can also swap e.g. Windows key for Command simply enough.
