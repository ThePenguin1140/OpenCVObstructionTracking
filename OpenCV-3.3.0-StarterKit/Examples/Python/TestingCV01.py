
print("TestingCV01: Program Started.")

import os
import platform
import shutil
import sys

if platform.architecture() != ("32bit", "WindowsPE"):
    print("ERROR: The OpenCV Starter Kit is targeted at 32-bit Python running on Windows (32-bit or 64-bit)!")
    sys.exit(1)

print("TestingCV01: Architecture Test Passed.")

# Make sure that NumPy is installed.
try:
    import numpy
except:
    print("ERROR: NumPy is required! Launch the command prompt with admin rights, change to your Python installation directory (e.g. \"CD C:\\Python27\") and enter: \"Scripts\\pip install numpy\". If you don't have the \"pip\" tool yet (Python's package manager), install it by entering: \"python -m ensurepip --upgrade\".")
    sys.exit(1)

print("TestingCV01: NumPy Installation Checked.")

# Copy all relevant OpenCV DLLs and the Python module into the current directory.
OPENCV_DLL_DIRECTORY = "../../OpenCV/x86/vc15/bin"
for filename in os.listdir(OPENCV_DLL_DIRECTORY):
    if (filename.endswith(".pyd") or filename.endswith("330.dll")) and not os.path.exists(filename):
        shutil.copy(os.path.join(OPENCV_DLL_DIRECTORY, filename), ".")

# Now that the OpenCV Python module is in the current directory, import it.
import cv2

print("TestingCV01: Imported cv2.")

# Callback function for mouse interaction.
def mouseEvent(evt, x, y, flags, frame):
    if evt == cv2.EVENT_LBUTTONDOWN:
        # Get the pixel that was clicked and print its color.
        color = frame[y, x]
        red = color[2]
        green = color[1]
        blue = color[0]
        sys.stdout.write("Pixel at (%d, %d) has color (%d, %d, %d).\n" % (x, y, red, green, blue))

# Create an image of size 320x240 with 3 channels of 8-bit unsigned integers.
# Draw a red anti-aliased line of thickness 3 from (10, 10) to (200, 100).
# Draw a filled green ellipse with center (160, 120), width 100, height 50 and angle 20°.
# Then, display the image in a window.
image = numpy.zeros((240, 320, 3), numpy.uint8)
cv2.line(image, (10, 10), (200, 100), (0, 0, 255), 3, cv2.LINE_AA)
cv2.ellipse(image, ((160, 120), (100, 50), 20), (0, 255, 0), -1, cv2.LINE_AA)
cv2.imshow("Hello World", image)

# Load some image from a file.
koalaImage = cv2.imread("../Koala.jpg")

# Initialize video capture from camera and check if it worked. If not, use a video file.
vidCap = cv2.VideoCapture(0)
if vidCap.isOpened():
    print("Successfully opened a camera.")

    # Some webcams return a strange image the first time.
    # So we just read one frame and ignore it.
    vidCap.read()
else:
    print("Could not open camera! Opening a video file instead ...")
    vidCap = cv2.VideoCapture("../Bunny.mp4")
    if not vidCap.isOpened():
        print("Could not open video file!")
        cv2.destroyAllWindows()
        sys.exit(1)

# Create an image with the correct size for the video frames and 3 channels of 8-bit unsigned integers.
video_width = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("TestingCV01: Video frame size is %dx%d pixels." % (video_width, video_height))
frame = numpy.zeros((video_height, video_width, 3), numpy.uint8)
result, frame = vidCap.read()

# Create another window and give it a name.
windowName = "Video"
cv2.namedWindow(windowName)

# Set the mouse interaction callback function for the window.
# The image matrix will be passed as a parameter.
cv2.setMouseCallback(windowName, mouseEvent, frame)

# The main loop.
while True:
    # Read a video frame.
    # We abort when have reached the end of the video stream.
    result, new_frame = vidCap.read()
    if not result:
        break

    # Make sure the image is a 3-channel 24-bit image.
    if frame.dtype != "uint8" or frame.shape[2] != 3:
        print("Unexpected image format!")
        sys.exit(1)

    # Copy the frame into our image.
    numpy.copyto(frame, new_frame)

    # Apply a 5x5 median filter.
    cv2.medianBlur(frame, 5, frame)

    # We will add the other image to our camera image.
    # If its size is not the same as the camera frame, resize it (this will only happen once).
    if koalaImage.shape[:2] != frame.shape[:2]:
        koalaImage = cv2.resize(koalaImage, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    cv2.addWeighted(frame, 0.75, koalaImage, 0.25, 0, frame)

    # Display a text.
    cv2.putText(frame, "Click somewhere!", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    # Set the pixel at x = 40, y = 20 to red.
    frame[20, 40] = (0, 0, 255)

    # Show the image in the window.
    cv2.imshow(windowName, frame)

    # Quit the loop when the Esc key is pressed.
    # Calling waitKey is important, even if you're not interested in keyboard input!
    keyPressed = cv2.waitKey(1)
    if keyPressed != -1:
        # Only the least-significant 16 bits contain the actual key code. The other bits contain modifier key states.
        keyPressed &= 0xFFFF
        sys.stdout.write("TestingCV01: Key pressed: %d\n" % keyPressed)
        if keyPressed == 27:
            break

# Clean up.
vidCap.release()
cv2.destroyAllWindows()

print("TestingCV01: Program Ended.")

