 #include <iostream>
 #include <DGtal/base/Common.h>
 #include <DGtal/io/readers/GenericReader.h>
 #include <DGtal/io/writers/RawWriter.h>
 #include <DGtal/helpers/StdDefs.h>
 #include <DGtal/images/Image.h>
 #include <DGtal/images/ImageContainerBySTLVector.h>
 
 #include "CLI11.hpp"
 
 
 using namespace std;
 using namespace DGtal;
 using namespace Z3i;
 
 
 void missingParam ( std::string param )
 {
   trace.error() <<" Parameter: "<<param<<" is required..";
   trace.info() <<std::endl;
   exit ( 1 );
 }
 
 
 int main(int argc, char**argv)
 {
 
   // parse command line using CLI ----------------------------------------------
   CLI::App app;
   std::string inputFileName;
   std::string outputFileName {"result.raw"};
   DGtal::int64_t rescaleInputMin {0};
   DGtal::int64_t rescaleInputMax {255};
   
   app.description("Convert a vol to a 8-bit raw file.\n Example: vol2raw  ${DGtal}/examples/samples/lobster.vol res.raw \n");
   app.add_option("-i,--input,1", inputFileName, "vol file (.vol, .longvol .p3d, .pgm3d and if WITH_ITK is selected: dicom, dcm, mha, mhd). For longvol, dicom, dcm, mha or mhd formats, the input values are linearly scaled between 0 and 255." )
     ->required()
     ->check(CLI::ExistingFile);
   app.add_option("--output,-o,2",outputFileName ,"output file (.raw).");
   app.add_option("--rescaleInputMin", rescaleInputMin, "min value used to rescale the input intensity (to avoid basic cast into 8  bits image).", true);
   app.add_option("--rescaleInputMax", rescaleInputMax, "max value used to rescale the input intensity (to avoid basic cast into 8  bits image).", true);
  
   
   app.get_formatter()->column_width(40);
   CLI11_PARSE(app, argc, argv);
   // END parse command line using CLI ----------------------------------------------
 
   
   typedef ImageContainerBySTLVector<Z3i::Domain, unsigned char>  MyImageC;
   typedef DGtal::functors::Rescaling<DGtal::int64_t ,unsigned char > RescalFCT;
   MyImageC imageC =  GenericReader< MyImageC >::importWithValueFunctor( inputFileName ,RescalFCT(rescaleInputMin,
                                                                                                  rescaleInputMax,
                                                                                                  0, 255) );
 
   bool res =  RawWriter< MyImageC >::exportRaw8(outputFileName, imageC);
   trace.info() << "Raw export done, image dimensions: "  << imageC.domain().upperBound()[0]-imageC.domain().lowerBound()[0]+1
                << " " << imageC.domain().upperBound()[1]-imageC.domain().lowerBound()[1]+1
                << " " << imageC.domain().upperBound()[2]-imageC.domain().lowerBound()[2]+1 << std::endl;
     
   if (res)
      return EXIT_SUCCESS;
   else
     return EXIT_FAILURE;
 }