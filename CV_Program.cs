using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV.UI;

namespace ShaprCVTest 
{
  class CV_Program 
  {
    public static bool       TrackCups    = false;
    public static bool       ShowMinYLine = false;
    public static CV_Cup[]   Cups                ;
    public static bool       ShowHSV      = false;
    public static bool       ShowFiltered = false;
    public static int        MinY         = 100  ;
    public static float      MinWidth     = 100  ;
    public static float      MinHeight    = 100  ;
    public static float      AvgHeight    = 100  ;
    public static float      AvgWidth     = 100  ;

    private static Capture   _capture     = null ;
    private static bool running           = false;

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

    public static void DetectCups_Webcam(bool ShowHSV = false, bool ShowFiltered = false)
    {
      CV_Program.ShowHSV = ShowHSV;
      CV_Program.ShowFiltered = ShowFiltered;
      int FrameRate = 30;

      ImageViewer view = new ImageViewer();
      _capture = new Capture();
      Application.Idle += new EventHandler(delegate(object sender, EventArgs args)
      {
        var frame = _capture.QueryFrame();
        if (!running)
        {
          //InitCupTracking(frame);
          running = !running;
        }
        else
        {
          try
          {
            view.Image = DetectCups(frame);
          }
          catch (Exception e)
          {
            //TODO jwuertz Need to make this more stable
            //Right now it freezes on error, I'd like it to skip frames when 
            //there's a problem.
            Console.Write("ERROR - SKIPPING FRAME: ");
            Console.WriteLine(e.Message);
            return;
          }
        }
        int keyPressed = CvInvoke.WaitKey( 1 );
        if (keyPressed != -1 && keyPressed != 255)
        {
          // Only the least-significant 16 bits contain the actual key code. The other bits contain modifier key states
          keyPressed &= 0xFFFF;
          if (keyPressed == 27) view.Close();
          else if (keyPressed == 116) InitCupTracking(frame);
          else if (keyPressed == 121) ShowMinYLine = true;
        }
      });

      view.ShowDialog();
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

      bool paused = false;

      // This matrix will contain our image.
      Mat frame = new Mat();
      double frameNum = vidCap.GetCaptureProperty( CapProp.PosFrames );
      double nextFrame = frameNum;
      frame = vidCap.QueryFrame();

      //The Main Loop: Instead of while(true)
      for ( int TimeOut = 0; TimeOut< 10000; TimeOut++ ) {
        // Read a video frame into our image
        // If we get an empty frame, we abort because have reached the end of the video stream
        if ( frame.IsEmpty ) break;

        // Make sure the image is a 3-channel 24-bit image
        if ( !( frame.Depth == DepthType.Cv8U ) && frame.NumberOfChannels == 3 ) {
          Console.WriteLine( "CV_Program: DetectCups_Video(): Unexpected image format!" );
          Console.WriteLine( "CV_Program: DetectCups_Video(): Type [" + frame.GetType().ToString() + "] and Channels [" + frame.NumberOfChannels + "]" );
          return;
        }

        // Show the image in the window
        CvInvoke.Imshow( wiName, DetectCups(frame) );

        // Quit the loop when the Esc key is pressed.
        // Calling waitKey is important, even if you're not interested in keyboard input!
        int keyPressed = CvInvoke.WaitKey( 1 );
        nextFrame = vidCap.GetCaptureProperty( CapProp.PosFrames );
        if ( paused ) nextFrame = vidCap.GetCaptureProperty( CapProp.PosFrames ) - 1;

        if ( keyPressed != -1 && keyPressed != 255 ) {
            // Only the least-significant 16 bits contain the actual key code. The other bits contain modifier key states
            keyPressed &= 0xFFFF;
//            Console.WriteLine( "CV_Program: DetectCups_Video(): Key pressed: " + keyPressed );
          if ( keyPressed == 27 ) break;
          else if ( keyPressed == 116 ) InitCupTracking( frame );
          else if ( keyPressed == 121 ) ShowMinYLine = true;
          else if ( keyPressed == 97 && nextFrame > 0 && paused ) {
            //prev 
            nextFrame -= 1;
          } else if ( keyPressed == 100 && nextFrame < vidCap.GetCaptureProperty( CapProp.FrameCount ) && paused ) {
            //next
            nextFrame += 1;
          } else if ( keyPressed == 32 ) paused = !paused;
        }
        vidCap.SetCaptureProperty( CapProp.PosFrames, nextFrame );
        frame = vidCap.QueryFrame();
      }

      Console.WriteLine( "CV_Program: DetectCups_Video(): Ended Video Function." );
    }

    public static Image<Bgr, byte> DetectCups( Mat input_image )
    {
    	Size size = new Size( 700, 700 );

      Image<Gray, byte> filtered_image = new Image<Gray, byte>( size );

      Image<Hsv, byte> preprocessed_image = PreProcessing.Preprocess( input_image, size, ShowHSV );

      //denoise, smoothe and threshold
      filtered_image = PreProcessing.FilterCups( preprocessed_image, ShowFiltered );

      //PreProcessing.FilterGlare( preprocessed_image, ShowFiltered );

    	Image<Bgr, byte> output_image = new Image<Bgr, byte>( input_image.Size );
    	output_image = input_image.ToImage<Bgr, byte>();
    	CvInvoke.Resize( output_image, output_image, size );

      PreProcessing.DrawContours( output_image, PreProcessing.GetContours( filtered_image ), filtered_image );

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
        Image<Gray, byte> filtered_image = new Image<Gray, byte>( size );

        //denoise, smoothe and threshold
        filtered_image = PreProcessing.FilterCups( PreProcessing.Preprocess( input_image, size, false ), false );

        VectorOfVectorOfPoint contours = PreProcessing.GetContours( filtered_image );

        for ( int i = 0; i < contours.Size; i++ ) 
        {
          if ( i < 3 ) Cups[i].Init(i,CvInvoke.BoundingRectangle( contours[i] ));
          else Console.WriteLine( "CV_Program: Too many contours during InitCupTracking()" );
        }

        Console.WriteLine( "CV_Program: InitCupTracking(): Found These Cups:" );
        Console.WriteLine( "CV_Program: InitCupTracking(): [" + Cups[0].CupID + "] " + Cups[0].BoundingBox);
        Console.WriteLine( "CV_Program: InitCupTracking(): [" + Cups[1].CupID + "] " + Cups[1].BoundingBox);
        Console.WriteLine( "CV_Program: InitCupTracking(): [" + Cups[2].CupID + "] " + Cups[2].BoundingBox);

        MinY = ((Cups[0].BoundingBox.Location.Y + Cups[1].BoundingBox.Location.Y + Cups[2].BoundingBox.Location.Y) / 3) - 80;

        float[] widths = new float[3];
        widths[0] = (float) Cups[0].BoundingBox.Width;
        widths[1] = (float) Cups[1].BoundingBox.Width;
        widths[2] = (float) Cups[2].BoundingBox.Width;
        
        float[] heights = new float[3];
        heights[0] = (float) Cups[0].BoundingBox.Height;
        heights[1] = (float) Cups[1].BoundingBox.Height;
        heights[2] = (float) Cups[2].BoundingBox.Height;
        
        MinWidth = getMaxWidthOrHeight(widths);
        MinWidth *= 1.3f;
        
        MinHeight = getMaxWidthOrHeight(heights);
        MinHeight *= 1.15f;
        
        AvgHeight = (Cups[0].BoundingBox.Height + Cups[1].BoundingBox.Height + Cups[2].BoundingBox.Height) / 3.0f;
        AvgWidth  = (Cups[0].BoundingBox.Width  + Cups[1].BoundingBox.Width  + Cups[2].BoundingBox.Width ) / 3.0f;

        TrackCups = true;
      }
    }

    private static float getMaxWidthOrHeight(float[] cupLengths)
    {
      float max = 0;

      for (int i = 0; i < cupLengths.Length; i++)
      {
        if (cupLengths[i] > max) max = cupLengths[i];
      }

      return max;
    }

    private static void DrawContours( Image<Bgr, byte> output, VectorOfVectorOfPoint contours, Mat frame = null ) {
      Bgr bgrRed = new Bgr( Color.Red );

      int boxID = 0;

      for ( int i = 0; i < contours.Size; i++ ) {
        Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

        output.Draw( box, bgrRed, 2 );

        if ( frame != null )
          CvInvoke.PutText( frame, "[" + ( i + 1 ) + "]", new System.Drawing.Point( box.Location.X + 5, box.Location.Y - 10 ), FontFace.HersheyPlain, 1.25, new MCvScalar( 255, 0, 255 ), 2 );

      }
    }
  }
}
