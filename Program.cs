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
    //Set to TRUE to display debugging messages on console
    static bool DebugMode = true;

    public static void Main()
    {
      DetectCups ();
      //SimpleVideoFeed ();
    }

    public static void DetectCups()
    {
      string windowName = "Cup Detector";
      Size size = new Size( 700, 700 );

      //Load image
      Mat input_image = CvInvoke.Imread( "..\\..\\Images\\Cups.jpg", LoadImageType.AnyColor );
      Image<Gray, byte> preprocessed_image = new Image<Gray, byte>( size );

      //denoise, smoothe and threshold
      Preprocess( input_image, preprocessed_image, size );

      Image<Bgr, byte> output_image = new Image<Bgr, byte>( input_image.Size );
      output_image = input_image.ToImage<Bgr, byte>();
      CvInvoke.Resize( output_image, output_image, size );
      DrawContours( output_image, GetContours( preprocessed_image ) );

      CvInvoke.Imshow( "threshold", preprocessed_image );
      CvInvoke.Imshow( windowName, output_image );
      //CvInvoke.Imshow(windowName, contours_image);
      //Wait for the key pressing event
      //Destroy the window if key is pressed
      CvInvoke.WaitKey( 0 );
      CvInvoke.DestroyWindow( windowName );
    }

    private static void CheckMouseClicks() 
    {
        System.Windows.Forms.MouseButtons Click = System.Windows.Forms.Control.MouseButtons;

        if ( Click != System.Windows.Forms.MouseButtons.None )
        {
          if (DebugMode) Console.WriteLine( "CheckMouseClicks(): Button Pressed = " + Click );
        }
    }

    private static void SimpleVideoFeed () 
    {
      
	    // Initialize video capture from camera and check if it worked. If not, use a video file.
      Capture vidCap = new Capture ( 0 );

      if ( vidCap.Grab())
      {
        if (DebugMode) Console.WriteLine( "SimpleVideoFeed(): Successfully opened a camera." );

		    // Some webcams return a strange image the first time.
		    // So we just read one frame and ignore it.
        vidCap.Retrieve( new Mat() );
	    }
      else
      {
		    if (DebugMode) Console.WriteLine( "SimpleVideoFeed(): Could not open camera!" );
        return;
	    }

	    // Just for fun, output the video frame size.
	    if (DebugMode) Console.WriteLine( "SimpleVideoFeed(): Video frame size is [ " + vidCap.Width + " x " + vidCap.Height + "] pixels." );

	    // Create a window and give it a name.
	    string wiName = "This is a Video";
	    CvInvoke.NamedWindow( wiName );

	    // This matrix will contain our image.
	    Mat frame = new Mat();

	    //The Main Loop: Instead of while(true)
	    for ( int TimeOut = 0; TimeOut < 10000; TimeOut++ )
      {
		    // Read a video frame into our image.
		    // If we get an empty frame, we abort because have reached the end of the video stream.
        vidCap.Retrieve( frame );
		    if ( frame.IsEmpty ) break;

		    // Make sure the image is a 3-channel 24-bit image.
		    if ( !( frame.Depth == DepthType.Cv8U ) && frame.NumberOfChannels == 3 )
        {
			    if (DebugMode) Console.WriteLine( "SimpleVideoFeed(): Unexpected image format!" );
			    if (DebugMode) Console.WriteLine( "SimpleVideoFeed(): Type [" + frame.GetType().ToString() + "] and Channels [" + frame.NumberOfChannels + "]" );
			    return;
		    }

		    // Apply a 5x5 median filter.
		    //CvInvoke.MedianBlur( frame, frame, 5 );

		    // Display a text.
		    //CvInvoke.PutText( frame, "Click somewhere!", new System.Drawing.Point( 50, 50 ), FontFace.HersheyPlain, 1.5, new MCvScalar( 255, 0, 255 ), 2 );

		    // Show the image in the window.
		    CvInvoke.Imshow( wiName, frame );

    		// Quit the loop when the Esc key is pressed.
    		// Calling waitKey is important, even if you're not interested in keyboard input!
    		int keyPressed = CvInvoke.WaitKey( 1 );

        CheckMouseClicks ();

    		if ( keyPressed != -1 && keyPressed != 255 )
        {
    			// Only the least-significant 16 bits contain the actual key code. The other bits contain modifier key states.
    			keyPressed &= 0xFFFF;
    			if (DebugMode) Console.WriteLine( "SimpleVideoFeed(): Key pressed: " + keyPressed );
    			if ( keyPressed == 27 ) break;
    		}
	    }

      if (DebugMode) Console.WriteLine( "SimpleVideoFeed(): Ended Video Function." );
    }

    private static VectorOfVectorOfPoint GetContours( Image<Gray, byte> input ) {

      VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();

      Image<Gray, float> laplace_image = input.Laplace( 3 );
      Image<Gray, float> erode_image = laplace_image.Erode( 2 );
      Image<Gray, byte> byteErode_image = erode_image.Convert<Gray, byte>();
      Image<Gray, byte> thresholded_image = byteErode_image.ThresholdToZero( new Gray( 240 ) );
      Image<Gray, byte> erode2_image = thresholded_image.Erode( 3 );

      int[,] tree = CvInvoke.FindContourTree( erode2_image, contours, ChainApproxMethod.ChainApproxSimple );

      for ( int n = 0; n < tree.Length / 4; n++ ) {
        for ( int m = 0; m < 4; m++ ) {
          Console.Write( tree[n, m] + ", " );
        }
        Console.WriteLine();
      }
      return contours;
    }

    private static void DrawContours( Image<Bgr, byte> output, VectorOfVectorOfPoint contours ) {
      Bgr bgrRed = new Bgr( Color.Red );

      for ( int i = 0; i < contours.Size; i++ ) {
        Rectangle box = CvInvoke.BoundingRectangle( contours[i] );
        if ( ( box.Width < 400 && box.Height < 400 ) &&
          ( box.Width > 50 && box.Height > 50 ) ) {
          output.Draw( box, bgrRed, 2 );
        }
        //output.Draw( box, bgrRed, 2 );
      }
    }

    private static void Preprocess( Mat input, Image<Gray, byte> output, Size size ) {

      //Resize image
      Image<Bgr, byte> resized_image = new Image<Bgr, byte>( size );
      CvInvoke.Resize( input, resized_image, size );

      CvInvoke.FastNlMeansDenoisingColored( resized_image, resized_image, 3, 3, 7, 21 );
      resized_image = resized_image.SmoothMedian( 15 );
      resized_image._GammaCorrect( 2d );
      resized_image._EqualizeHist();

      Image<Hsv, byte> hsv_image = new Image<Hsv, byte>( size );
      CvInvoke.CvtColor( resized_image, hsv_image, ColorConversion.Bgr2Hsv );

      CvInvoke.Imshow( "hsv", hsv_image );

      Image<Gray, byte> gray_image = new Image<Gray, byte>( size );
      CvInvoke.CvtColor( resized_image, gray_image, ColorConversion.Bgr2Gray );

      CvInvoke.Imshow( "gray", gray_image );

      ScalarArray lower = new ScalarArray( new Hsv( 0, 0, 0 ).MCvScalar );
      ScalarArray upper = new ScalarArray( new Hsv( 35, 255, 255 ).MCvScalar );

      CvInvoke.InRange( hsv_image, lower, upper, output );

      //CvInvoke.AdaptiveThreshold( gray_image, output, 255, AdaptiveThresholdType.GaussianC, ThresholdType.Binary, 15, 4 );
    }
  }
}
