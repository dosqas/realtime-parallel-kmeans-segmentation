#include "video_io.hpp"
#include "coreset.hpp"
#include <opencv2/opencv.hpp>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

	// Start webcam feed
    showWebcamFeed();

	MPI_Finalize();

    return 0;
}