#include "video_io.hpp"
#include "coreset.hpp"
#include <opencv2/opencv.hpp>
#include <mpi.h>

// Entry point
int main(int argc, char** argv) 
{
	// Initialize our MPI environment
	MPI_Init(&argc, &argv);

	// Prevent OpenCV from inserting logs in the console
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		std::cout << "Starting application..." << std::endl << std::endl;
		std::cout << "Press 'Esc' to quit." << std::endl;
		std::cout << "Press '1' for the sequential backend." << std::endl;
		std::cout << "Press '2' for the distributed MPI/OpenMP hybrid backend." << std::endl;
		std::cout << "Press '3' for the multithreaded backend" << std::endl;
		std::cout << "Press '4' for the CUDA backend (if available)." << std::endl << std::endl;
		std::cout << "Hint: use the 'k' slider to adjust the number of segments.";
	}

	// Display the webcam feed using OpenCV
	showWebcamFeed();

	// Finalizes and cleans the MPI environment.
	MPI_Finalize();

	if (rank == 0) {
		std::cout << std::endl << std::endl << std::endl << "Shutting down..." << std::endl;
	}

    return 0;
}