using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace ShaprCVTest
{
	class MainClass
	{
		public static void Main ()
		{
			DetectCups();
		}

		public static void DetectCups()
		{
			string windowName = "Cup Detector";
			Size size = new Size(500, 500);

			//Load image
			Mat input_image = CvInvoke.Imread("..\\..\\Images\\Cups.jpg", LoadImageType.AnyColor);
			Image<Hsv, byte> preprocessed_image = new Image<Hsv, byte>(size);
			Preprocess(input_image, preprocessed_image, size);
			//blur the image to reduce noise and smoothe edges
			CvInvoke.GaussianBlur(preprocessed_image, preprocessed_image, new Size(31, 31), 0, 0);


			ScalarArray lower = new ScalarArray(new Hsv(0, 0, 0).MCvScalar);
			ScalarArray upper = new ScalarArray(new Hsv(90, 255, 255).MCvScalar);
			Image<Gray, byte> filtered_image = new Image<Gray, byte>(size);
			CvInvoke.InRange(preprocessed_image, lower, upper, filtered_image);

			Image<Bgr, byte> output_image = new Image<Bgr, byte>(input_image.Size);
			output_image = input_image.ToImage<Bgr, byte>();
			CvInvoke.Resize(output_image, output_image, size);
			DrawBoundingBoxes(filtered_image, output_image);

			CvInvoke.Imshow(windowName, output_image);
			//CvInvoke.Imshow(windowName, contours_image);
			//Wait for the key pressing event
			//Destroy the window if key is pressed
			CvInvoke.WaitKey(0);
			CvInvoke.DestroyWindow(windowName);

		}

		private static void DrawBoundingBoxes(Image<Gray, byte> input, Image<Bgr, byte> output)
		{
			Image<Gray, float> laplace_image = input.Laplace(3);
			Image<Gray, float> erode_image = laplace_image.Erode(2);
			Image<Gray, byte> byteErode_image = erode_image.Convert<Gray, byte>();
			Image<Gray, byte> thresholded_image = byteErode_image.ThresholdToZero(new Gray(240));
			Image<Gray, byte> erode2_image = thresholded_image.Erode(3);

			VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
			Mat tree = new Mat();

			CvInvoke.FindContours(erode2_image, contours, tree, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);

			Console.Write(tree);

			Bgr bgrRed = new Bgr(Color.Red);

			for (int i = 0; i < contours.Size; i++)
			{
				Rectangle box = CvInvoke.BoundingRectangle(contours[i]);
				if ((box.Width < 400 && box.Height < 400) &&
					(box.Width > 20 && box.Height > 20))
				{
					output.Draw(box, bgrRed, 2);
				}
			}
		}

		private static void Preprocess(Mat input, Image<Hsv, byte> output, Size size) { 

			//Resize image
			Image<Bgr, byte> resized_image = new Image<Bgr, byte>(size);
			CvInvoke.Resize(input, resized_image, size);
			CvInvoke.CvtColor(resized_image, output, ColorConversion.Bgr2Hsv);

		}
	}
}
