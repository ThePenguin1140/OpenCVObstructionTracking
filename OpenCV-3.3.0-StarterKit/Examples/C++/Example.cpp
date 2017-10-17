// Include header files.
#include <iostream>
#include <opencv2/opencv.hpp>

// Callback function for mouse interaction.
void mouseEvent(int evt,
				int x,
				int y,
				int flags,
				void* p_param)
{
	cv::Mat& frame = *reinterpret_cast<cv::Mat*>(p_param);

	if (evt == cv::EVENT_LBUTTONDOWN)
	{
		// Get the pixel that was clicked and print its color.
		cv::Vec3b color = frame.at<cv::Vec3b>(y, x);
		int red = color(2);
		int green = color(1);
		int blue = color(0);
		std::cout << "Pixel at (" << x << ", " << y << ") has color (" << red << ", " << green << ", " << blue << ")." << std::endl;
	}
}

int main(int argc,
		 char* argv[])
{
	// Create an image of size 320x240 with 3 channels of 8-bit unsigned integers.
	// Draw a red anti-aliased line of thickness 3 from (10, 10) to (200, 100).
	// Draw a filled green ellipse with center (160, 128), width 100, height 50 and angle 20°.
	// Then, display the image in a window.
	cv::Mat image(cv::Size(320, 240), CV_8UC3);
	// Alternatively, for a typed matrix: cv::Mat3b image(cv::Size(320, 240));
	cv::line(image, cv::Point(10, 10), cv::Point(200, 100), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	cv::ellipse(image, cv::RotatedRect(cv::Point(160, 120), cv::Size(100, 50), 20.0f), cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
	cv::imshow("Hello World", image);

	// Load some image from a file.
	cv::Mat koalaImage = cv::imread("../Koala.jpg");

	// Initialize video capture from camera and check if it worked. If not, use a video file.
	cv::VideoCapture vidCap = cv::VideoCapture(0);
	if (vidCap.isOpened())
	{
		std::cout << "Successfully opened a camera." << std::endl;

		// Some webcams return a strange image the first time.
		// So we just read one frame and ignore it.
		vidCap >> cv::Mat();
	}
	else
	{
		std::cout << "Could not open camera! Opening a video file instead ..." << std::endl;
		vidCap = cv::VideoCapture("../Bunny.mp4");
		if (!vidCap.isOpened())
		{
			std::cout << "Could not open video file!" << std::endl;
			return 1;
		}
	}

	// Just for fun, output the video frame size.
	std::cout << "Video frame size is " << static_cast<int>(vidCap.get(cv::CAP_PROP_FRAME_WIDTH)) << "x" << static_cast<int>(vidCap.get(cv::CAP_PROP_FRAME_HEIGHT)) << " pixels." << std::endl;

	// Create another window and give it a name.
	std::string windowName = "Video";
	cv::namedWindow(windowName);

	// This matrix will contain our image.
	cv::Mat frame;

	// Set the mouse interaction callback function for the window.
	// The image matrix will be passed as a parameter.
	cv::setMouseCallback(windowName, &mouseEvent, &frame);

	// The main loop.
	while (true)
	{
		// Read a video frame into our image.
		// If we get an empty frame, we abort because have reached the end of the video stream.
		vidCap >> frame;
		if (frame.empty())
			break;

		// Make sure the image is a 3-channel 24-bit image.
		if (frame.type() != CV_8UC3)
		{
			std::cout << "Unexpected image format!" << std::endl;
			return 1;
		}

		// Apply a 5x5 median filter.
		cv::medianBlur(frame, frame, 5);

		// We will add the other image to our camera image.
		// If its size is not the same as the camera frame, resize it (this will only happen once).
		if (koalaImage.size() != frame.size()) cv::resize(koalaImage, koalaImage, frame.size(), 0, 0, cv::INTER_CUBIC);
		frame = 0.75 * frame + 0.25 * koalaImage;

		// Display a text.
		cv::putText(frame, "Click somewhere!", cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 0, 255), 2);

		// Set the pixel at x = 40, y = 20 to red.
		frame.at<cv::Vec3b>(20, 40) = cv::Vec3b(0, 0, 255);

		// Show the image in the window.
		cv::imshow(windowName, frame);

		// Quit the loop when the Esc key is pressed.
		// Calling waitKey is important, even if you're not interested in keyboard input!
		int keyPressed = cv::waitKey(1);
		if (keyPressed != -1)
		{
			// Only the least-significant 16 bits contain the actual key code. The other bits contain modifier key states.
			keyPressed &= 0xFFFF;
			std::cout << "Key pressed: " << keyPressed << std::endl;
			if (keyPressed == 27)
				break;
		}
	}

	std::cout << "That's it!" << std::endl;

	return 0;
}