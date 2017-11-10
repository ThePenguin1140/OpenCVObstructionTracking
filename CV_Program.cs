using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.Features2D;
using Emgu.CV.XFeatures2D;
using System.Drawing;

namespace ShaprCVTest 
{
  class CV_Program 
  {
    public static bool ShowHSV  = false;
    public static bool ShowGray = false;

    private static void ExtractFeatures( Mat modelImg, Mat obsImg, out VectorOfKeyPoint modelKey, out VectorOfKeyPoint observedKey, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography ) {
      int k = 2;
      double uniqueThresh = 0.5;
      double hessianThres = 300;

      homography = null;
      mask = null;

      modelKey = new VectorOfKeyPoint();
      observedKey = new VectorOfKeyPoint();

      using ( UMat uModel = modelImg.ToUMat( AccessType.Read ) )
      using ( UMat uObs = obsImg.ToUMat( AccessType.Read ) ) {
        SURF surfCPU = new SURF( hessianThres );

        UMat modelDescriptors = new UMat();
        surfCPU.DetectAndCompute( uModel, null, modelKey, modelDescriptors, false );

        UMat obsDescriptors = new UMat();
        surfCPU.DetectAndCompute( uObs, null, observedKey, obsDescriptors, false );

        BFMatcher matcher = new BFMatcher( DistanceType.L2 );
        matcher.Add( modelDescriptors );

        matcher.KnnMatch( obsDescriptors, matches, k, null );
        mask = new Mat( matches.Size, 1, DepthType.Cv8U, 1 );
        mask.SetTo( new MCvScalar( 255 ) );
        Features2DToolbox.VoteForUniqueness( matches, uniqueThresh, mask );

        int nonZeroCount = CvInvoke.CountNonZero( mask );
        if ( nonZeroCount >= 4 ) {
          nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation( modelKey, observedKey, matches, mask, 1.5, 20 );
          if ( nonZeroCount >= 4 ) {
            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures( modelKey, observedKey, matches, mask, 2 );
          }
        }
      }
    }

    public static void SURFDetectCups( string inputImagePath = "..//..//Images//Cups.jpg" ) {

      Size size = new Size( 700, 700 );

      Mat leftCorner = SURFPreprocessing( CvInvoke.Imread( "..\\..\\Images\\cups_bottom_left.jpg", LoadImageType.AnyColor ), size );
      Mat rightCorner = SURFPreprocessing( CvInvoke.Imread( "..\\..\\Images\\cups_bottom_right.jpg", LoadImageType.AnyColor ), size );
      Mat test = SURFPreprocessing( CvInvoke.Imread( "..\\..\\Images\\cups_bottom.jpg", LoadImageType.AnyColor ), size );

      Mat inputImage = CvInvoke.Imread( inputImagePath, LoadImageType.AnyColor );

      Mat preprocessedImg = SURFPreprocessing( inputImage, size );

      CvInvoke.Imshow( "Draw SURF", DrawSURF( test, leftCorner ) );

      CvInvoke.WaitKey( 0 );
      CvInvoke.DestroyAllWindows();
    }

    private static Mat DrawSURF( Mat modelImage, Mat observedImage ) {
      Mat homography;
      VectorOfKeyPoint modelKeyPoints, observedKeyPoints;

      using ( VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch() ) {
        Mat mask;
        ExtractFeatures( modelImage, observedImage, out modelKeyPoints, out observedKeyPoints, matches, out mask, out homography );

        Mat result = new Mat();
        Features2DToolbox.DrawMatches( modelImage, modelKeyPoints, observedImage, observedKeyPoints, matches, result, new MCvScalar( 255, 255, 255 ), new MCvScalar( 255, 255, 255 ), mask );

        if ( homography != null ) {
          Rectangle rect = new Rectangle( Point.Empty, modelImage.Size );
          PointF[] pts = new PointF[]
          {
            new PointF( rect.Left,   rect.Bottom),
            new PointF( rect.Right,  rect.Bottom),
            new PointF( rect.Left,   rect.Top),
            new PointF( rect.Right,  rect.Top)
          };
          pts = CvInvoke.PerspectiveTransform( pts, homography );

          Point[] points = Array.ConvertAll<PointF, Point>( pts, Point.Round );
          using ( VectorOfPoint vp = new VectorOfPoint( points ) ) {
            CvInvoke.Polylines( result, vp, true, new MCvScalar( 255, 0, 0, 255 ), 5 );
          }
        }

        return result;
      }
    }

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

    public static void DetectCups_Video( string VidPath = "..\\..\\Videos\\Cups.mp4", bool ShowHSV = false, bool ShowGray = false) 
    {
      Console.WriteLine( "CV_Program: DetectCups_Video(): [" + ShowHSV + ", " + ShowGray + "] " + VidPath + "" );

      CV_Program.ShowHSV  = ShowHSV ;
      CV_Program.ShowGray = ShowGray;

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
          if ( keyPressed == 27 ) break;
        }
      }

      Console.WriteLine( "CV_Program: DetectCups_Video(): Ended Video Function." );
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

      DrawContours( output_image, GetContours( preprocessed_image ), output_image.Mat );

    	return output_image;
    }

    private static VectorOfVectorOfPoint GetContours( Image<Gray, byte> input )
    {
    	VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
      VectorOfVectorOfPoint contour2 = new VectorOfVectorOfPoint();

    	Image<Gray, float> laplace_image      = input.Laplace( 3 );
    	Image<Gray, float> erode_image        = laplace_image.Erode( 2 );
    	Image<Gray, byte > byteErode_image    = erode_image.Convert<Gray, byte>();
    	Image<Gray, byte > thresholded_image  = byteErode_image.ThresholdToZero( new Gray( 240 ) );
    	Image<Gray, byte > erode2_image       = thresholded_image.Erode( 3 );

    	int[,] tree = CvInvoke.FindContourTree( erode2_image, contours, ChainApproxMethod.ChainApproxSimple );
      int[,] tre2 = new int[tree.Length / 4, 4];

      int t2id = 0;

      //Filter Countors: remove oddly sized ones
      for ( int i = 0; i<contours.Size; i++ )
      {
        Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

        if ( ( box.Width  < 400       && box.Height     < 400 ) &&
             ( box.Width  > 50        && box.Height     > 50  ) &&
             ( box.Height > box.Width && box.Location.Y > 100 ) ) 
             {
                tre2[t2id, 0] = tree[i, 0];
                tre2[t2id, 1] = tree[i, 1];
                tre2[t2id, 2] = tree[i, 2];
                tre2[t2id, 3] = tree[i, 3];
                contour2.Push(contours[i]);
                t2id++;
             }
      }

      Console.WriteLine("\nCV_Program: GetControus():\n");

    	for ( int n = 0; n < t2id; n++ )
      {
    		for ( int m = 0; m < 4; m++ )
        {
          Console.Write( tre2[n, m].ToString().PadLeft(5));
    		}

    		Console.WriteLine();
    	}

    	return contour2;
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
                CvInvoke.PutText( frame, "[" + i+1 + "]", new System.Drawing.Point( box.Location.X + 5, box.Location.Y - 10 ), FontFace.HersheyPlain, 1.25, new MCvScalar( 255, 0, 255 ), 2 );
             
    	}
    }

    private static Mat SURFPreprocessing( Mat input, Size size ) {
      //Resize image
      Image<Bgr, byte> resized_image = new Image<Bgr, byte>( size );
      CvInvoke.Resize( input, resized_image, size );

      //Causes a lot of lag between frames
      //CvInvoke.FastNlMeansDenoisingColored( resized_image, resized_image, 3, 3, 7, 21 );

      //Causes a bit of lag between frames
      resized_image = resized_image.SmoothGaussian( 15 );
      resized_image._GammaCorrect( 2d );
      resized_image._EqualizeHist();

      Mat output = new Mat();
      CvInvoke.CvtColor( resized_image, output, ColorConversion.Bgr2Hsv );

      return output;
    }


    private static void Preprocess( Mat input, Image<Gray, byte> output, Size size )
    {

    	//Resize image
    	Image<Bgr, byte> resized_image = new Image<Bgr, byte>( size );
    	CvInvoke.Resize( input, resized_image, size );

      //Causes a lot of lag between frames
    	//CvInvoke.FastNlMeansDenoisingColored( resized_image, resized_image, 3, 3, 7, 21 );

      //Causes a bit of lag between frames
      resized_image = resized_image.SmoothGaussian( 15 );
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
