#include <iostream>
#include <string>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkNormalizeImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkImageFileWriter.h"

#include <iomanip>

// Extracts the name of a image from the filepath
std::string extractName(const std::string& filename) {
	size_t lastindex = filename.find_first_of(".");
	size_t firstindex = filename.find_last_of("/") + 1;

	if ((firstindex == std::string::npos) || (lastindex == std::string::npos)) return filename;

	std::string temp = filename.substr(0, lastindex);
	return temp.substr(firstindex, lastindex);
}

int main(int argc, char *argv[])
{
	if(argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
		return EXIT_FAILURE;
	}
	
	std::string sourcename = extractName(argv[1]);
	
	typedef itk::Image<double, 3> FloatImageType;

	typedef itk::ImageFileReader<FloatImageType>
		ReaderType;
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(argv[1]);

	typedef itk::NormalizeImageFilter< FloatImageType, FloatImageType >
		NormalizeFilterType;
	NormalizeFilterType::Pointer normalizeFilter = NormalizeFilterType::New();
	normalizeFilter->SetInput(reader->GetOutput());

	typedef itk::StatisticsImageFilter< FloatImageType >
		StatisticsFilterType;
	StatisticsFilterType::Pointer statistics1 = StatisticsFilterType::New();
	statistics1->SetInput(reader->GetOutput());

	StatisticsFilterType::Pointer statistics2 = StatisticsFilterType::New();
	statistics2->SetInput(normalizeFilter->GetOutput());

	std::stringstream desc1;
	statistics1->Update();
	desc1 << itksys::SystemTools::GetFilenameName(argv[1])
		<< "\nMean: " << statistics1->GetMean()
		<< " Variance: " << statistics1->GetVariance();
	std::cout << itksys::SystemTools::GetFilenameName(argv[1])
		<< "\nMean: " << statistics1->GetMean()
		<< " Variance: " << statistics1->GetVariance() << '\n';


	std::stringstream desc2;
	statistics2->Update();
	desc2 << "Normalize"
		<< "\nMean: "
		<< std::fixed << std::setprecision (2) << statistics2->GetMean()
		<< " Variance: " << statistics2->GetVariance();
	std::cout << "Normalize"
		<< "\nMean: "
		<< std::fixed << std::setprecision (2) << statistics2->GetMean()
		<< " Variance: " << statistics2->GetVariance() << '\n';


	
	std::string outputname = sourcename + "_normalized.nii.gz";	

	typedef itk::ImageFileWriter<FloatImageType> WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetInput(normalizeFilter->GetOutput());
	writer->SetFileName(outputname);
	std::cout << "Writing Normalized Image\n";
	writer->Update();
	return EXIT_SUCCESS;
}
