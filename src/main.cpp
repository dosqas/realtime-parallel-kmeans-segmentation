#include "video_io.hpp"
#include "coreset.hpp"
#include <opencv2/opencv.hpp>
#include <mpi.h>

// Entry point
int main(int argc, char** argv) 
{
	// Prevent OpenCV from inserting logs in the console
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	// Initialize our MPI environment
    MPI_Init(&argc, &argv);

	// Display the webcam feed using OpenCV
    showWebcamFeed();

	// Finalizes and cleans the MPI environment.
	MPI_Finalize();

    return 0;
}