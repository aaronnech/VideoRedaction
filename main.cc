#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>

using namespace cv;
using namespace std;


/**
 * Applies a "blur" to a particular image, where a "blur" is defined to be
 * the effect of detecting all rectangles found by the given classifier and
 * applying a blurring effect to those rectangles
 * @param image      the given image
 * @param imageGrey  a greyscale counterpart of the image: provides better classifier results
 * @param classifier the classifier doing the classification
 */
static void applyBlurToDetected(Mat &image, Mat &imageGrey, CascadeClassifier &classifier) {
    vector< Rect_<int> > rectanglesDetected;
    classifier.detectMultiScale(imageGrey, rectanglesDetected);

    for (int i = 0; i < rectanglesDetected.size(); i++) {
        // Process each found rectangular area of interest
        Rect found = rectanglesDetected[i];

        // Get area as a matrix
        Mat area = image(found);

        // Blur this area
        Mat blurred;
        blur(area, blurred, Size(30, 30));

        // Copy blurred segement back to detected area
        blurred.copyTo(area);
    }
}

/**
 * Processes a chunk of the video
 * @param frames            The buffer vector holding the video frames
 * @param start             The start frame
 * @param end               The end frame
 * @param faceClassifier    The face classifier
 * @param profileClassifier The profile classifier
 */
static void processFrameChunk(
        vector<Mat> &frames,
        int start, int end,
        CascadeClassifier &faceClassifier,
        CascadeClassifier &profileClassifier) {
    // TODO: modify parameters to take more classifiers as a list?

    cout << "Processing chunk from " << start << " to " << end << endl;

    for (int i = start; i < end; i++) {
        Mat frame = frames[i];

        cout << "frame " << i << " -- Turning grey for processing" << endl;

        // Turn to greyscale
        Mat grey;
        cvtColor(frame, grey, CV_BGR2GRAY);

        cout << "frame " << i << " -- Applying blur" << endl;

        // Apply blur
        boost::thread blurFaces(applyBlurToDetected, frame, grey, faceClassifier);
        boost::thread blurProfiles(applyBlurToDetected, frame, grey, profileClassifier);

        blurFaces.join();
        blurProfiles.join();

        cout << "frame " << i << " done processing" << endl;
    }
}

/**
 * Gets the frames of a video
 * @return the frames of a video as a vector containing matrix elements
 */
static vector<Mat> getFrames(VideoCapture &video) {
    vector<Mat> result;
    int currentFrame = 0;
    while (true) {
        Mat frame;
        video >> frame;
        if (frame.empty()) break;
        result.push_back(frame);
        cout << "Frame " << (currentFrame) << " loaded" << endl;
        currentFrame++;
    }

    return result;
}

/**
 * Prints the usage documentation of the program and exits
 */
static void usage(char* programName) {
    cout << "Usage: " << endl;
    cout << programName << " infile outfile cores" << endl;
    cout << "\tCores is the number of parallel threads activated for processing video frames. It must be >0" << endl;
    cout << "\tUse \"-\" for outfile to display video when finished processing instead" << endl;
    exit(1);
}

int main(int argc, char** argv) {
    // Process command line arguments
    if (argc != 4) usage(argv[0]);

    int coreCount = atoi(argv[3]);
    if (coreCount <= 0) usage(argv[0]);

    string inFilePath(argv[1]);
    string outFilePath(argv[2]);


    // Open video
    VideoCapture inputVideo(inFilePath);
    if(!inputVideo.isOpened())
        return -1;

    Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),
                  (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
    cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
         << " of nr#: " << inputVideo.get(CV_CAP_PROP_FRAME_COUNT) << endl;


    // Create classifiers, unfortunately we need one classifier per parallel chunk
    CascadeClassifier profileClassifier[coreCount];
    CascadeClassifier faceClassifier[coreCount];

    VideoWriter outputVideo;

    // Create window if we are displaying the output, otherwise create an output video
    if (outFilePath == "-") {
        namedWindow("Face Blur", 1);
    } else {
        // Output video
        cout << inputVideo.get(CV_CAP_PROP_FPS) << endl;
        int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));
        outputVideo.open(outFilePath, ex, inputVideo.get(CV_CAP_PROP_FPS), S, true);
    }

    cout << "Loading video into memory..." << endl;
    vector<Mat> frames = getFrames(inputVideo);

    cout << "Loaded " << frames.size() << " frames" << endl;

    cout << "Processing frames..." << endl;

    // Process each frame
    
    // Use a thread pool to watch over all the threads
    boost::thread_group group;
    for (int i = 0; i < coreCount; i++) {
        // Calculate chunk bounds
        int start = i * (frames.size() / coreCount);
        int end = ((i + 1) * (frames.size() / coreCount)) - 1;

        // Load classifier models (one for each thread)
        profileClassifier[i].load("haarcascade_profileface.xml");
        faceClassifier[i].load("haarcascade_frontalface_default.xml");

        // Kick off the thread
        boost::thread *processFrameJob = new boost::thread(processFrameChunk, frames, start, end, profileClassifier[i], faceClassifier[i]);
        group.add_thread(processFrameJob);
    }

    // Will destruct thread objects
    group.join_all();

    cout << "Outputing video..." << endl;

    // Replay the video if we're playing in a window
    do {
        for (int i = 0; i < frames.size(); i++) {
            if (outFilePath == "-") {
                imshow("Face Blur", frames[i]);
                waitKey(30);
            } else {
                // stream to output file
                outputVideo << frames[i];
            }
        }     
    } while(outFilePath == "-");


    return 0;
}