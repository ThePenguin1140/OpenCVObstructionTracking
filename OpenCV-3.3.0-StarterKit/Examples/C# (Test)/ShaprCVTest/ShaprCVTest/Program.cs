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
			DetectCups();
		}

		public static void DetectCups()
		{
			string windowName = "Cup Detector";


			//Load image
			Mat input_image = CvInvoke.Imread("..\\..\\Images\\Cups.jpg", LoadImageType.AnyColor);

			//Resize image
			Size size = new Size(500, 500);
			Image<Bgr, Byte> resized_image = new Image<Bgr, Byte>(size);
			CvInvoke.Resize(input_image, resized_image, size);


			CvInvoke.GaussianBlur(resized_image, resized_image, new Size(31, 31), 0, 0);
			Image<Hsv, Byte> hsv_image = new Image<Hsv, Byte>(size);
			Image<Gray, Byte> filtered_image = new Image<Gray, Byte>(size);

			CvInvoke.CvtColor(resized_image, hsv_image, ColorConversion.Bgr2Hsv);

			ScalarArray lower = new ScalarArray(new Hsv(0, 0, 0).MCvScalar);
			ScalarArray upper = new ScalarArray(new Hsv(90, 255, 255).MCvScalar);


			CvInvoke.InRange(hsv_image, lower, upper, filtered_image);


			Image<Gray, float> laplace_image = filtered_image.Laplace(3);
			Image<Gray, float> erode_image = laplace_image.Erode(2);
			Image<Gray, Byte> byteErode_image = erode_image.Convert<Gray, Byte>();
			Image<Gray, Byte> thresholded_image = byteErode_image.ThresholdToZero(new Gray(240));
			Image<Gray, Byte> erode2_image = thresholded_image.Erode(3);

			VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
			Mat tree = new Mat();

			CvInvoke.FindContours(erode2_image, contours, tree, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);

			Console.Write(tree);

			Bgr bgrRed = new Bgr(Color.Red);
			Image<Bgr, Byte> output_image = new Image<Bgr, Byte>(size);

			CvInvoke.CvtColor(filtered_image, output_image, ColorConversion.Gray2Bgr);

			for (int i = 0; i < contours.Size; i++)
			{
				Rectangle box = CvInvoke.BoundingRectangle(contours[i]);
				if ((box.Width < 400 && box.Height < 400) &&
					(box.Width > 20 && box.Height > 20))
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
	}
}
