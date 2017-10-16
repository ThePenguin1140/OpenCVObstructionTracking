using System;
using Emgu.Util;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Collections.Generic;
using Emgu.CV.Util;
using System.Drawing;

namespace ShaprCVTest
{
	class MainClass
	{
		public static void Main (string[] args)
		{
			//CreateWindow 		("This Is Window Name");
			//DrawExampleStuff 	();
			//VideoExampleStuff	();
			DetectCups();
		}

		public static void DetectCups()
		{
			string windowName = "Cup Detector";



			//Mat image = CvInvoke.Imread("C:\\Users\\jwuertz\\Documents\\GitHub\\OpenCVObstructionTracking\\OpenCV-3.3.0-StarterKit\\Examples\\C# (Test)\\ShaprCVTest\\ShaprCVTest\\Images\\Cups.jpg");
			Mat input_image = CvInvoke.Imread("..\\..\\Images\\Cups.jpg", LoadImageType.AnyColor);
			Image<Bgr, Byte> resized_image = new Image<Bgr, Byte>(500, 500);
			System.Drawing.Size size = new System.Drawing.Size(500, 500);
			CvInvoke.Resize(input_image, resized_image, size);
			CvInvoke.GaussianBlur(resized_image, resized_image, new System.Drawing.Size(31, 31),0,0);
			Image<Hsv, Byte> hsv_image = new Image<Hsv, Byte>(500, 500);
			Image<Gray, Byte> filtered_image = new Image<Gray, Byte>(500, 500);

			CvInvoke.CvtColor(resized_image, hsv_image, ColorConversion.Bgr2Hsv);

			ScalarArray lower = new ScalarArray(new Hsv(0,0,0).MCvScalar);
			ScalarArray upper = new ScalarArray(new Hsv(90,255,255).MCvScalar);


			CvInvoke.InRange(hsv_image, lower, upper, filtered_image);


			Image<Gray,float> laplace_image = filtered_image.Laplace(3);
			Image<Gray, float> erode_image = laplace_image.Erode(2);
			Image<Gray, Byte> byteErode_image = erode_image.Convert<Gray, Byte>();
			Image<Gray, Byte> thresholded_image = byteErode_image.ThresholdToZero(new Gray(240));
			Image<Gray, Byte> erode2_image = thresholded_image.Erode(3);

			VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
			Mat tree = new Mat();
			
			CvInvoke.FindContours(erode2_image, contours, tree, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);

			Console.Write(tree);

			Bgr bgrRed = new Bgr(Color.Red);
			Image<Bgr, Byte> output_image = new Image<Bgr, Byte>(filtered_image.Size);

			CvInvoke.CvtColor(filtered_image, output_image, ColorConversion.Gray2Bgr);

			for (int i = 0; i < contours.Size; i++) 
			{
				Rectangle box = CvInvoke.BoundingRectangle(contours[i]);
				if ( ( box.Width < 400 && box.Height < 400 ) &&
				    ( box.Width > 20 && box.Height > 20 ) )
				{
					output_image.Draw(box, bgrRed, 2);
				}
			}

			CvInvoke.Imshow(windowName, output_image);
			//CvInvoke.Imshow(windowName, contours_image);
			//Wait for the key pressing event
			//Destroy the window if key is pressed
			CvInvoke.WaitKey(0);
			CvInvoke.DestroyWindow(windowName);

		}
		/*
		public static void CreateWindow (string WindowName)
		{ 
			CvInvoke.NamedWindow (WindowName); 					//Create the window using the specific name

			Mat img = new Mat(200, 400, DepthType.Cv8U, 3); 	//Create a 3 channel image of 400x200
			img.SetTo(new Bgr(255, 0, 0).MCvScalar); 			// set it to Blue color

			CvInvoke.Imshow (WindowName, img); 					//Show the image

			CvInvoke.WaitKey (0);  								//Wait for the key pressing event
			CvInvoke.DestroyWindow (WindowName); 				//Destroy the window if key is pressed
		}

		private static void DrawExampleStuff ()
		{
			string wiName = "Line & Ellipse!";

			// Create an image of size 320x240 with 3 channels of 8-bit unsigned integers.
			Mat image = new Mat(320, 240, DepthType.Cv8U, 3);

			// Draw a red anti-aliased line of thickness 3 from (10, 10) to (200, 100).
			CvInvoke.Line(image, new System.Drawing.Point(10, 10), new System.Drawing.Point(200, 100), new MCvScalar(0, 0, 255), 3, LineType.AntiAlias);

			// Draw a filled green ellipse with center (160, 128), width 100, height 50 and angle 20°.
			CvInvoke.Ellipse(image, new RotatedRect	(new System.Drawing.Point(160, 120), new System.Drawing.Size(100, 50), 20.0f),
							 		new MCvScalar	(0, 255, 0), -1, LineType.AntiAlias);

			// Then, display the image in a win.
			CvInvoke.Imshow(wiName, image);

			//Wait for the key pressing event
			//Destroy the window if key is pressed
			CvInvoke.WaitKey (0);
			CvInvoke.DestroyWindow (wiName);
		}

		private static void VideoExampleStuff ()
		{
			// Load some image from a file.
			Mat koalaImage = CvInvoke.Imread ("..\\..\\External Media\\Koala.jpg");

			// Initialize video capture from camera and check if it worked. If not, use a video file.
			VideoCapture vidCap = new VideoCapture(0);
			if (vidCap.IsOpened)
			{
				Console.WriteLine("ShaprCVTest: Successfully opened a camera.");

				// Some webcams return a strange image the first time.
				// So we just read one frame and ignore it.
				vidCap.Read(new Mat());
			}
			else
			{
				Console.WriteLine("ShaprCVTest: Could not open camera! Opening a video file instead ...");

				vidCap = new VideoCapture ("..\\..\\External Media\\Bunny.mp4");

				if (!vidCap.IsOpened)
				{
					Console.WriteLine("ShaprCVTest: Could not open video file!");
					return;
				}
			}

			// Just for fun, output the video frame size.
			Console.WriteLine("ShaprCVTest: Video frame size is [ " + vidCap.Width + " x " + vidCap.Height + "] pixels.");

			// Create another window and give it a name.
			string wiName = "This is a Video";
			CvInvoke.NamedWindow(wiName);

			// This matrix will contain our image.
			Mat frame = new Mat();



			//The Main Loop: Instead of while(true)
			for (int TimeOut = 0; TimeOut < 10000; TimeOut++)
			{
				// Read a video frame into our image.
				// If we get an empty frame, we abort because have reached the end of the video stream.
				vidCap.Read (frame); 
				if (frame.IsEmpty) break;

				// Make sure the image is a 3-channel 24-bit image.
				if (!(frame.Depth == DepthType.Cv8U) && frame.NumberOfChannels == 3)
				{
					Console.WriteLine("ShaprCVTest: Unexpected image format!");
					Console.WriteLine("ShaprCVTest: Type [" + frame.GetType().ToString() +
					                  "] and Channels [" + frame.NumberOfChannels + "]");
					return;
				}

				// Apply a 5x5 median filter.
				CvInvoke.MedianBlur (frame, frame, 5);

				// We will add the other image to our camera image.
				// If its size is not the same as the camera frame, resize it (this will only happen once).
				if (koalaImage.Size != frame.Size)
					CvInvoke.Resize (koalaImage, koalaImage, frame.Size, 0, 0, interpolation:Inter.Cubic);

				CvInvoke.AddWeighted (frame, 0.75f, koalaImage, 0.25f, 0.0f, frame);

				// Display a text.
				CvInvoke.PutText (frame, "Click somewhere!", new System.Drawing.Point(50, 50), FontFace.HersheyPlain, 1.5, new MCvScalar(255, 0, 255), 2);

				// Set the pixel at x = 40, y = 20 to red.
				//frame.at<cv::Vec3b>(20, 40) = cv::Vec3b(0, 0, 255);
				//Couldn't find a straight forward way to do this yet

				// Show the image in the window.
				CvInvoke.Imshow (wiName, frame);

				// Quit the loop when the Esc key is pressed.
				// Calling waitKey is important, even if you're not interested in keyboard input!
				int keyPressed = CvInvoke.WaitKey (1);
			
				if (keyPressed != -1 && keyPressed != 255)
				{
					// Only the least-significant 16 bits contain the actual key code. The other bits contain modifier key states.
					keyPressed &= 0xFFFF;
					Console.WriteLine("ShaprCVTest: Key pressed: " + keyPressed);
					if (keyPressed == 27) break;
				}

			}


			/*
				//THIS WAS DONE BEFORE LOOP
				// Set the mouse interaction callback function for the window.
				// The image matrix will be passed as a parameter.
				cv::setMouseCallback(windowName, &mouseEvent, &frame);
			 */
		/*
			Console.WriteLine("ShaprCVTest: Ended Video Function.");
		}
	*/
	}
}
