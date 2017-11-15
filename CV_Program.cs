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
    public static bool      TrackCups = false;
    public static CV_Cup[]  Cups             ;
    public static bool ShowHSV  = false;
    public static bool ShowFiltered = false;

    public static void DetectCups_Image( string ImgPath = "..\\..\\Images\\Cups.jpg", bool ShowHSV = false, bool ShowGray = false) 
    {
      Console.WriteLine( "CV_Program: DetectCups_Image(): [" + ShowHSV + ", " + ShowGray + "] " + ImgPath + "" );

      CV_Program.ShowHSV  = ShowHSV ;
      CV_Program.ShowFiltered = ShowGray;

      Mat frame = CvInvoke.Imread( ImgPath, LoadImageType.AnyColor );

      CvInvoke.Imshow( "DetectCups_Image", DetectCups( frame ) );

      Console.WriteLine( "\nCV_Program: DetectCups_Image(): Done, waiting for [Any] key" );

      CvInvoke.WaitKey( 0 );
      CvInvoke.DestroyAllWindows();
    }

    public static void DetectCups_Video( string VidPath = "..\\..\\Videos\\Cups.mp4", bool ShowHSV = false, bool ShowGray = false) 
    {
      Console.WriteLine( "CV_Program: DetectCups_Video(): [" + ShowHSV + ", " + ShowGray + "] " + VidPath + "" );

      CV_Program.ShowHSV  = ShowHSV ;
      CV_Program.ShowFiltered = ShowGray;

      // Initialize video capture from the video file and check if it worked.
      Capture vidCap = new Capture( VidPath );

      if ( vidCap.Grab())
      {
        Console.WriteLine( "CV_Program: DetectCups_Video(): Successfully opened video file" );
      }
      else
      {
        Console.WriteLine( "CV_Program: DetectCups_Video(): Failed to open video file" );
        return;
      }

      // Create a window and give it a name.
      string wiName = "DetectCups_Video";
      CvInvoke.NamedWindow( wiName );

      // This matrix will contain our image.
      Mat frame = new Mat();

      //The Main Loop: Instead of while(true)
      for ( int TimeOut = 0; TimeOut< 10000; TimeOut++ )
      {
        // Read a video frame into our image
        // If we get an empty frame, we abort because have reached the end of the video stream
        vidCap.Grab();
        vidCap.Retrieve( frame );
        if ( frame.IsEmpty ) break;

        // Make sure the image is a 3-channel 24-bit image
        if ( !( frame.Depth == DepthType.Cv8U ) && frame.NumberOfChannels == 3 )
        {
          Console.WriteLine( "CV_Program: DetectCups_Video(): Unexpected image format!" );
          Console.WriteLine( "CV_Program: DetectCups_Video(): Type [" + frame.GetType().ToString() + "] and Channels [" + frame.NumberOfChannels + "]" );
          return;
        }

        // Show the image in the window
        CvInvoke.Imshow( wiName, DetectCups(frame) );

        // Quit the loop when the Esc key is pressed.
        // Calling waitKey is important, even if you're not interested in keyboard input!
        int keyPressed = CvInvoke.WaitKey( 1 );

        if ( keyPressed != -1 && keyPressed != 255 )
        {
          // Only the least-significant 16 bits contain the actual key code. The other bits contain modifier key states
          keyPressed &= 0xFFFF;
          Console.WriteLine( "CV_Program: DetectCups_Video(): Key pressed: " + keyPressed );
          if ( keyPressed == 27  ) break;
          if ( keyPressed == 116 ) InitCupTracking(frame);
        }

      }

      Console.WriteLine( "CV_Program: DetectCups_Video(): Ended Video Function." );
    }

    public static Image<Bgr, byte> DetectCups( Mat input_image )
    {
    	Size size = new Size( 700, 700 );

      Image<Gray, byte> filtered_image = new Image<Gray, byte>( size );

      Image<Hsv, byte> preprocessed_image = Preprocess( input_image, size );

      //denoise, smoothe and threshold
      filtered_image = FilterCups( preprocessed_image );

      FilterGlare( preprocessed_image );

    	Image<Bgr, byte> output_image = new Image<Bgr, byte>( input_image.Size );
    	output_image = input_image.ToImage<Bgr, byte>();
    	CvInvoke.Resize( output_image, output_image, size );

      DrawContours( output_image, GetContours( filtered_image ), output_image.Mat );

    	return output_image;
    }

    public static void InitCupTracking( Mat input_image )
    {
      if ( !TrackCups )
      {
        Cups = new CV_Cup[3];

        for ( int i = 0; i < 3; i++ )
        {
          Cups[i] = new CV_Cup();
        }

        Size size = new Size( 700, 700 );

        //Load image
        Image<Gray, byte> preprocessed_image = new Image<Gray, byte>( size );

        //denoise, smoothe and threshold
        PreProcessing.Preprocess( input_image, preprocessed_image, size );

        VectorOfVectorOfPoint contours = PreProcessing.GetContours( preprocessed_image );

        for ( int i = 0; i < contours.Size; i++ ) 
        {
          if ( i < 3 ) Cups[i].Init(i,CvInvoke.BoundingRectangle( contours[i] ));
          else Console.WriteLine( "CV_Program: Too many contours during InitCupTracking()" );
        }

        Console.WriteLine( "CV_Program: InitCupTracking(): Found These Cups:" );
        Console.WriteLine( "CV_Program: InitCupTracking(): [" + Cups[0].CupID + "] " + Cups[0].BoundingBox);
        Console.WriteLine( "CV_Program: InitCupTracking(): [" + Cups[1].CupID + "] " + Cups[1].BoundingBox);
        Console.WriteLine( "CV_Program: InitCupTracking(): [" + Cups[2].CupID + "] " + Cups[2].BoundingBox);

        TrackCups = true;
      }
    }

    private static void DrawContours( Image<Bgr, byte> output, VectorOfVectorOfPoint contours, Mat frame = null )
    {
    	Bgr bgrRed = new Bgr( Color.Red );

      int boxID = 0;

    	for ( int i = 0; i < contours.Size; i++ )
      {
    		Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

        output.Draw( box, bgrRed, 2 );

        if ( frame != null )
            CvInvoke.PutText( frame, "[" + (i+1) + "]", new System.Drawing.Point( box.Location.X + 5, box.Location.Y - 10 ), FontFace.HersheyPlain, 1.25, new MCvScalar( 255, 0, 255 ), 2 );
             
    	}
    }


    private static Image<Hsv, byte> Preprocess( Mat input, Size size )
    {

    	//Resize image
      Image<Bgr, byte> resized_image = new Image<Bgr, byte>( size );
    	CvInvoke.Resize( input, resized_image, size );

      //Causes a lot of lag between frames
    	//CvInvoke.FastNlMeansDenoisingColored( resized_image, resized_image, 3, 3, 7, 21 );

      //Causes a bit of lag between frames
      resized_image = resized_image.SmoothGaussian( 15 );
    	resized_image._GammaCorrect( 2.5 );
    	resized_image._EqualizeHist();

      resized_image = resized_image.Erode( 10 );
      resized_image = resized_image.Dilate( 10 );

    	Image<Hsv, byte> hsv_image = new Image<Hsv, byte>( size );
    	CvInvoke.CvtColor( resized_image, hsv_image, ColorConversion.Bgr2Hsv );

      Image<Hsv,byte> output = new Image<Hsv, byte>( size );
    	CvInvoke.CvtColor( resized_image, output, ColorConversion.Bgr2Hsv );

      if ( ShowHSV )
    	CvInvoke.Imshow( "hsv", output );

      return output;
    }

    private static Image<Gray, byte> FilterCups( Image<Hsv, byte> input ) {
      ScalarArray lower = new ScalarArray( new Hsv( 0, 0, 0 ).MCvScalar );
      ScalarArray upper = new ScalarArray( new Hsv( 35, 255, 255 ).MCvScalar );

      Image<Gray, byte> output = new Image<Gray, byte>( input.Size );

      CvInvoke.InRange( input, lower, upper, output );

      if ( ShowFiltered )
        CvInvoke.Imshow( "Cup Filter", output );

      return output;
    }

    private static Image<Gray, byte> FilterGlare( Image<Hsv, byte> input ) {
      Image<Gray, byte> output = new Image<Gray, byte>( input.Size );

      ScalarArray lower = new ScalarArray( new Hsv( 75, 0, 0 ).MCvScalar );
      ScalarArray upper = new ScalarArray( new Hsv( 180, 200, 255 ).MCvScalar );

      CvInvoke.InRange( input, lower, upper, output );

      if ( ShowFiltered )
        CvInvoke.Imshow( "Glare Filter", output );

      return output;
    }

  }
}
