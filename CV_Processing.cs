using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.Collections;

namespace ShaprCVTest
{
  class PreProcessing
  {
    //To keep track of which 'contours' need to be split
    //1 = no split, 2 = in half, etc.
    private static ArrayList    BoxSplits                         ;
    private static int[]        CupNumsFound  = new int[99]       ;
    private static Rectangle[]  NewBoxes      = new Rectangle[99] ;
    private static int          CNF_i         = 0                 ;


    public static bool HasParent( VectorOfVectorOfPoint currentContours, Rectangle checkMe )
    {
      for ( int n = 0; n < currentContours.Size; n++ )
      {
        Rectangle Box = CvInvoke.BoundingRectangle( currentContours[n] );

        if (  Box.Location.X < checkMe.Location.X &&
			        Box.Location.Y < checkMe.Location.Y &&
			        Box.Location.X + Box.Width  > checkMe.Location.X + checkMe.Width &&
			        Box.Location.Y + Box.Height > checkMe.Location.Y + checkMe.Width )

			        return true;
      }

      return false;
    }

    public static VectorOfVectorOfPoint GetContours( Image<Gray, byte> input )
    {
      VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
    	VectorOfVectorOfPoint contour2 = new VectorOfVectorOfPoint();

      BoxSplits = new ArrayList();


    	Image<Gray, float> laplace_image      = input.Laplace( 3 );
    	Image<Gray, float> erode_image        = laplace_image.Erode( 2 );
    	Image<Gray, byte>  byteErode_image    = erode_image.Convert<Gray, byte>();
    	Image<Gray, byte>  thresholded_image  = byteErode_image.ThresholdToZero( new Gray( 240 ) );
    	Image<Gray, byte>  erode2_image       = thresholded_image.Erode( 3 );

    	int[,] tree = CvInvoke.FindContourTree( erode2_image, contours, ChainApproxMethod.ChainApproxSimple );
    	int[,] tre2 = new int[tree.Length / 4, 4];

    	int t2id = 0;

	    //Filter Countors: remove oddly sized ones
	    for ( int i = 0; i < contours.Size; i++ )
      {
        Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

		    if ( ( box.Width < 400 && box.Height < 400 ) &&
			     ( box.Width > 50 && box.Height > 50 ) &&
			     ( box.Height > box.Width && box.Location.Y > 100 ) && !HasParent( contour2, CvInvoke.BoundingRectangle( contours[i] ) ) )
           {
              tre2[t2id, 0] = tree[i, 0];
        			tre2[t2id, 1] = tree[i, 1];
        			tre2[t2id, 2] = tree[i, 2];
        			tre2[t2id, 3] = tree[i, 3];

              if ( (float)box.Width / (float)box.Height > 0.52f )
              {
                BoxSplits.Add( 2 );
              } 
              else
              {
                BoxSplits.Add( 1 );
              }

              contour2.Push( contours[i] );
        			t2id++;
		      }
      }

      //Uncomment the following Console.WriteLines to display Bounding Box details
	    //Console.WriteLine( "\nCV_Program: GetControus():\n" );

	    for ( int n = 0; n < t2id; n++ )
      {
        for ( int m = 0; m < 4; m++ )
        {
          //Console.Write( tre2[n, m].ToString().PadLeft( 5 ) );
		    }
        //Console.WriteLine();
      }

      return contour2;
    }

    public static void DrawContours( Image<Bgr, byte> output, VectorOfVectorOfPoint contours, Mat frame = null )
    {
      Bgr bgrRed = new Bgr( Color.Red );

      ResetCupsFound();

	    for ( int i = 0; i < contours.Size; i++ ) 
      {
        Rectangle box = CvInvoke.BoundingRectangle( contours[i] );

        if ( !CV_Program.TrackCups || GetMinDistance(box) < 150 )
        {
          if ( (int)BoxSplits[i] == 1 ) 
          {
            output.Draw( box, bgrRed, 2 );
            if ( frame != null && CV_Program.TrackCups)
              CvInvoke.PutText( frame, "[" + ( GetCupNum( box) + 1 ) + "]", new System.Drawing.Point( box.Location.X + 5, box.Location.Y - 10 ), FontFace.HersheyPlain, 1.25, new MCvScalar( 255, 0, 255 ), 2 );
          } 
          else if ( (int)BoxSplits[i] == 2 ) 
          {
            Rectangle box1 = new Rectangle( box.X - 1, box.Y, box.Width / 2, box.Height );
            Rectangle box2 = new Rectangle( box.X + ( box.Width / 2 ) + 1, box.Y, box.Width / 2, box.Height );
            output.Draw( box1, bgrRed, 2 );
            output.Draw( box2, bgrRed, 2 );

            if ( frame != null && CV_Program.TrackCups )
              CvInvoke.PutText( frame, "[" + ( GetCupNum( box2 ) + 1 ) + "]", new System.Drawing.Point( box2.Location.X + 5, box2.Location.Y - 10 ), FontFace.HersheyPlain, 1.25, new MCvScalar( 255, 0, 255 ), 2 );
            if ( frame != null && CV_Program.TrackCups )
              CvInvoke.PutText( frame, "[" + ( GetCupNum( box1 ) + 1 ) + "]", new System.Drawing.Point( box1.Location.X + 5, box1.Location.Y - 10 ), FontFace.HersheyPlain, 1.25, new MCvScalar( 255, 0, 255 ), 2 );

          }  
        }
      }

      UpdateCupTracking();
    }

    public static void Preprocess( Mat input, Image<Gray, byte> output, Size size ) 
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

    	if ( CV_Program.ShowHSV )
    		CvInvoke.Imshow( "hsv", hsv_image );

    	Image<Gray, byte> gray_image = new Image<Gray, byte>( size );
    	CvInvoke.CvtColor( resized_image, gray_image, ColorConversion.Bgr2Gray );

    	ScalarArray lower = new ScalarArray( new Hsv( 0, 0, 0 ).MCvScalar );
    	ScalarArray upper = new ScalarArray( new Hsv( 35, 255, 255 ).MCvScalar );

      CvInvoke.InRange( hsv_image, lower, upper, output );
    }

    public static int GetCupNum( Rectangle Box )
    {
      int   n = -1;
      float d = -1.0f;

      for ( int i = 0; i < 3; i++ )
      {
        float newDist = GetDistance( Box, CV_Program.Cups[i].BoundingBox );
        if ( !IsIn(i) && ( newDist < d || d == -1.0f ) )
        {
          n = i;
          d = newDist;
        }
      }

      if ( n >= 0 )
      {
        NewBoxes[CNF_i] = Box;
        CupNumsFound[CNF_i] = n;
        CNF_i++;
      }

      return n;
    }

    public static float GetDistance( Rectangle rect1, Rectangle rect2 )
    {
      Point center1 = Center( rect1 );
      Point center2 = Center( rect2 );

      float horizontalDistance = Math.Abs(( center2.X - center1.X ));
      float verticalDistance   = Math.Abs(( center2.Y - center1.Y ));

      float distance = (float) Math.Sqrt( ( horizontalDistance * horizontalDistance ) + ( verticalDistance * verticalDistance ) );

	    return distance;
    }

    public static float GetMinDistance( Rectangle box )
    {
      float d = float.PositiveInfinity;
      if ( CV_Program.TrackCups )
      {
        for ( int i = 0; i< 3; i++ )
        {
          float newDist = GetDistance( box, CV_Program.Cups[i].BoundingBox );

          if ( newDist < d )
          {
            d = newDist;
          }
        }
      }

      return d;
    }

    public static Point Center( Rectangle rect )
    {
      return new Point( rect.Left + rect.Width / 2, rect.Top + rect.Height / 2);
    }

    private static void ResetCupsFound()
    {
      CupNumsFound = new int[99];
      NewBoxes        = new Rectangle[99];
      CNF_i = 0;
    }

    private static bool IsIn( int x )
    {
      for ( int i = 0; i < CNF_i; i++ )
      {
        if ( x == CupNumsFound[i] ) return true;
      }
      return false;
    }
  
    public static void UpdateCupTracking()
    {
      if ( CV_Program.TrackCups )
      {
        for ( int i = NewBoxes.Length-1; i >= 0; i-- )
        {
          for ( int j = 0; j < 3; j++ )
          {
            if ( CupNumsFound[i] == j ) CV_Program.Cups[j].BoundingBox = NewBoxes[i];
          }
        }
      }
    }

  }
}
