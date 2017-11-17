using System;
using System.Diagnostics;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.Net.Mime;
using System.Threading;
using System.Windows.Forms;
using Emgu.CV.UI;

namespace ShaprCVTest
{
    class CV_Program
    {
        public static bool TrackCups = false;
        public static CV_Cup[] Cups;
        public static bool ShowHSV = false;
        public static bool ShowFiltered = false;

        private static Capture vidCap;
        private static Image<Bgr, byte> frame = new Image<Bgr, byte>(new Size(0,0));

        private static bool running = false;
        private static double frameNum = 0d;
        private static Object lockObject = new Object();

        public static void DetectCups_Image(string ImgPath = "..\\..\\Images\\Cups.jpg", bool ShowHSV = false,
            bool ShowGray = false)
        {
            Console.WriteLine("CV_Program: DetectCups_Image(): [" + ShowHSV + ", " + ShowGray + "] " + ImgPath + "");

            CV_Program.ShowHSV = ShowHSV;
            CV_Program.ShowFiltered = ShowGray;

            Mat frame = CvInvoke.Imread(ImgPath, LoadImageType.AnyColor);

            CvInvoke.Imshow("DetectCups_Image", DetectCups(frame));

            Console.WriteLine("\nCV_Program: DetectCups_Image(): Done, waiting for [Any] key");

            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }

        public static void DetectCups_Video(string VidPath = "..\\..\\Videos\\Cups.mp4", bool ShowHSV = false,
            bool ShowGray = false)
        {
            double totalFrames = 0d;

            Console.WriteLine("CV_Program: DetectCups_Video(): [" + ShowHSV + ", " + ShowGray + "] " + VidPath + "");

            CV_Program.ShowHSV = ShowHSV;
            CV_Program.ShowFiltered = ShowGray;

            ImageViewer myView = new ImageViewer();
            myView.Size = new Size(700, 700);
            myView.AutoSizeMode = AutoSizeMode.GrowAndShrink;
            myView.Show();
//            CvInvoke.NamedWindow("Video");

            // Initialize video capture from the video file and check if it worked.
            try
            {
                vidCap = new Capture(VidPath);
                vidCap.ImageGrabbed += ProcessFrame;
                totalFrames = vidCap.GetCaptureProperty(CapProp.FrameCount);
                vidCap.Start();
                Console.WriteLine("Started capture");
                running = true;
            }
            catch (NullReferenceException e)
            {
                Console.WriteLine(e.Message);
            }

            while (running && totalFrames > frameNum)
            {
                lock (lockObject)
                {
                    if (frame != null && frame.Width > 0)
                    {
                        myView.Image = frame;
                        myView.Update();
                    }
                }
//                Thread.Sleep(500);
            }

            vidCap.Stop();
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }

        private static void ProcessFrame(object sender, EventArgs args)
        {
            double curFrame = 0d;
            var myInputFrame = new Mat();
            Image<Bgr, byte> myOutputFrame = null;

//            Capture _cap = (Capture) sender;
            try
            {
                lock (lockObject)
                {
                    curFrame = vidCap.GetCaptureProperty(CapProp.PosFrames);
                    vidCap.Retrieve(myInputFrame, 0);
                }
            }
            catch(Exception e)
            {
                Console.Write("Error in threaded frame capture: ");
                Console.WriteLine(e.Message);
                StackTrace st = new StackTrace(e, true);
                var frame = st.GetFrame(0);
                var line = frame.GetFileLineNumber();
                Console.WriteLine(line);
            }

            try
            {
                if (myInputFrame == null || myInputFrame.IsEmpty) return;

                //process myFrame
                myOutputFrame = DetectCups(myInputFrame);
            }
            catch (Exception e)
            {
                Console.Write("Error in threaded processing: ");
                Console.WriteLine(e.Message);
                var st = new StackTrace(e, true);
                var frame = st.GetFrame(0);
                var line = frame.GetFileLineNumber();
                Console.WriteLine(line);
            }
            
            try
            {
                lock (lockObject)
                {
                    if (curFrame > frameNum)
                    {
                        frameNum = curFrame;
                        frame = myOutputFrame;
                    }
                }
            }
            catch (Exception e)
            {
                Console.Write("Error in threaded frame update: ");
                Console.WriteLine(e.Message);
            }
        }

        public static Image<Bgr, byte> DetectCups(Mat input_image)
        {
            Size size = new Size(700, 700);

            Image<Gray, byte> filtered_image = new Image<Gray, byte>(size);

            Image<Hsv, byte> preprocessed_image = PreProcessing.Preprocess(input_image, size, ShowHSV);

            //denoise, smoothe and threshold
            filtered_image = PreProcessing.FilterCups(preprocessed_image, ShowFiltered);

            PreProcessing.FilterGlare(preprocessed_image, ShowFiltered);

            Image<Bgr, byte> output_image = new Image<Bgr, byte>(input_image.Size);
            output_image = input_image.ToImage<Bgr, byte>();
            CvInvoke.Resize(output_image, output_image, size);

            PreProcessing.DrawContours(output_image, PreProcessing.GetContours(filtered_image), output_image.Mat);

            return output_image;
        }

        public static void InitCupTracking(Mat input_image)
        {
            if (!TrackCups)
            {
                Cups = new CV_Cup[3];

                for (int i = 0; i < 3; i++)
                {
                    Cups[i] = new CV_Cup();
                }

                Size size = new Size(700, 700);

                //Load image
                Image<Gray, byte> filtered_image = new Image<Gray, byte>(size);

                //denoise, smoothe and threshold
                filtered_image = PreProcessing.FilterCups(PreProcessing.Preprocess(input_image, size, false), false);

                VectorOfVectorOfPoint contours = PreProcessing.GetContours(filtered_image);

                for (int i = 0; i < contours.Size; i++)
                {
                    if (i < 3) Cups[i].Init(i, CvInvoke.BoundingRectangle(contours[i]));
                    else Console.WriteLine("CV_Program: Too many contours during InitCupTracking()");
                }

                Console.WriteLine("CV_Program: InitCupTracking(): Found These Cups:");
                Console.WriteLine("CV_Program: InitCupTracking(): [" + Cups[0].CupID + "] " + Cups[0].BoundingBox);
                Console.WriteLine("CV_Program: InitCupTracking(): [" + Cups[1].CupID + "] " + Cups[1].BoundingBox);
                Console.WriteLine("CV_Program: InitCupTracking(): [" + Cups[2].CupID + "] " + Cups[2].BoundingBox);

                TrackCups = true;
            }
        }

        private static void DrawContours(Image<Bgr, byte> output, VectorOfVectorOfPoint contours, Mat frame = null)
        {
            Bgr bgrRed = new Bgr(Color.Red);

            int boxID = 0;

            for (int i = 0; i < contours.Size; i++)
            {
                Rectangle box = CvInvoke.BoundingRectangle(contours[i]);

                output.Draw(box, bgrRed, 2);

                if (frame != null)
                    CvInvoke.PutText(frame, "[" + (i + 1) + "]",
                        new System.Drawing.Point(box.Location.X + 5, box.Location.Y - 10), FontFace.HersheyPlain, 1.25,
                        new MCvScalar(255, 0, 255), 2);
            }
        }
    }
}