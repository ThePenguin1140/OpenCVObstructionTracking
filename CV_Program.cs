using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace ShaprCVTest 
{
  class CV_Program 
  {
    public static bool ShowHSV  = false;
    public static bool ShowGray = false;

    public static void DetectCups_Image( string ImgPath = "..\\..\\Images\\Cups.jpg", bool ShowHSV = false, bool ShowGray = false) 
    {
      Console.WriteLine( "CV_Program: DetectCups_Image(): [" + ShowHSV + ", " + ShowGray + "] " + ImgPath + "" );

      CV_Program.ShowHSV  = ShowHSV ;
      CV_Program.ShowGray = ShowGray;

      Mat frame = CvInvoke.Imread( ImgPath, LoadImageType.AnyColor );

      CvInvoke.Imshow( "DetectCups_Image", DetectCups( frame ) );

      Console.WriteLine( "\nCV_Program: DetectCups_Image(): Done, waiting for [Any] key" );

      CvInvoke.WaitKey( 0 );
      CvInvoke.DestroyAllWindows();
    }

    public static void DetectCups_Video( string VidPath = "..\\..\\Videos\\Cups.avi", bool ShowHSV = false, bool ShowGray = false) 
    {
      Console.WriteLine( "CV_Program: DetectCups_Video(): [" + ShowHSV + ", " + ShowGray + "] " + VidPath + "" );

      CV_Program.ShowHSV  = ShowHSV ;
      CV_Program.ShowGray = ShowGray;

      /*
       * 
       * INCOMPLETE ! ! ! 
       * 
       */
    }

    public static Image<Bgr, byte> DetectCups( Mat input_image )
    {
    	Size size = new Size( 700, 700 );

    	//Load image
    	Image<Gray, byte> preprocessed_image = new Image<Gray, byte>( size );

    	//denoise, smoothe and threshold
    	Preprocess( input_image, preprocessed_image, size );

    	Image<Bgr, byte> output_image = new Image<Bgr, byte>( input_image.Size );
    	output_image = input_image.ToImage<Bgr, byte>();
    	CvInvoke.Resize( output_image, output_image, size );

    	DrawContours( output_image, GetContours( preprocessed_image ) );

    	return output_image;
    }

    private static VectorOfVectorOfPoint GetContours( Image<Gray, byte> input )
    {
    	VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();

    	Image<Gray, float> laplace_image      = input.Laplace( 3 );
    	Image<Gray, float> erode_image        = laplace_image.Erode( 2 );
    	Image<Gray, byte > byteErode_image    = erode_image.Convert<Gray, byte>();
    	Image<Gray, byte > thresholded_image  = byteErode_image.ThresholdToZero( new Gray( 240 ) );
    	Image<Gray, byte > erode2_image       = thresholded_image.Erode( 3 );

    	int[,] tree = CvInvoke.FindContourTree( erode2_image, contours, ChainApproxMethod.ChainApproxSimple );

      Console.WriteLine("\nCV_Program: GetControus():\n");

    	for ( int n = 0; n < tree.Length / 4; n++ )
      {
    		for ( int m = 0; m < 4; m++ )
        {
          Console.Write( tree[n, m].ToString().PadLeft(5));
    		}

    		Console.WriteLine();
    	}

    	return contours;
    }

    private static void DrawContours( Image<Bgr, byte> output, VectorOfVectorOfPoint contours )
    {
    	Bgr bgrRed = new Bgr( Color.Red );

    	for ( int i = 0; i < contours.Size; i++ )
      {
    		Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

    		if ( ( box.Width  < 400 && box.Height < 400 ) &&
    			   ( box.Width  > 50  && box.Height > 50  ) &&
    			   ( box.Height > box.Width ) ) 
             {
                output.Draw( box, bgrRed, 2 );
    		     }
    	}
    }

    private static void Preprocess( Mat input, Image<Gray, byte> output, Size size )
    {

    	//Resize image
    	Image<Bgr, byte> resized_image = new Image<Bgr, byte>( size );
    	CvInvoke.Resize( input, resized_image, size );

    	CvInvoke.FastNlMeansDenoisingColored( resized_image, resized_image, 3, 3, 7, 21 );
    	resized_image = resized_image.SmoothMedian( 15 );
    	resized_image._GammaCorrect( 2d );
    	resized_image._EqualizeHist();

    	Image<Hsv, byte> hsv_image = new Image<Hsv, byte>( size );
    	CvInvoke.CvtColor( resized_image, hsv_image, ColorConversion.Bgr2Hsv );

      if ( ShowHSV )
    	CvInvoke.Imshow( "hsv", hsv_image );

    	Image<Gray, byte> gray_image = new Image<Gray, byte>( size );
    	CvInvoke.CvtColor( resized_image, gray_image, ColorConversion.Bgr2Gray );

      if ( ShowGray )
    	CvInvoke.Imshow( "gray", gray_image );

    	ScalarArray lower = new ScalarArray( new Hsv( 0 , 0  , 0   ).MCvScalar );
    	ScalarArray upper = new ScalarArray( new Hsv( 35, 255, 255 ).MCvScalar );

    	CvInvoke.InRange( hsv_image, lower, upper, output );
    }

  }
}
