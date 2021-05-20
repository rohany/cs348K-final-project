// This include is generated code by Halide, so the editor might
// not be happy to include it.
#include "portrait_gen.h"
#include "portrait_gen_auto.h"

#include <iostream>

#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "benchmark.h"

#include "args.hxx"

int main(int argc, char** argv) {

  args::ArgumentParser parser("Runs the portrait mode generator.", "");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

  args::ValueFlag<bool> useAuto(parser, "autoscheduled", "use the autoscheduled kernel", {'a', "auto"}, false);
  args::ValueFlag<bool> bench(parser, "bench", "benchmark", {'b', "bench"}, false);
  args::ValueFlag<bool> randomData(parser, "random", "use random inputs", {"random"}, false);

  args::ValueFlag<int> depthCutoff(parser, "depthCutoff", "cutoff for include as background for portrait mode", {"depth"}, 40);
  args::ValueFlag<std::string> left(parser, "left image", "left stereo image", {'l', "left"});
  args::ValueFlag<std::string> right(parser, "right image", "right stereo image", {'r', "right"});
  args::ValueFlag<std::string> seg(parser, "segmented left image", "segmented left image", {'s', "segmented"});
  args::ValueFlag<std::string> depth(parser, "depth map", "depth map image filename", {'d', "depth_map"});
  args::ValueFlag<std::string> output(parser, "output image", "output image filename", {'o', "output"});

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  } catch (args::ValidationError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  Halide::Runtime::Buffer<uint8_t> inLeft, inRight, segmented;
  if (args::get(randomData)) {
    // Load some random images.
    inLeft = Halide::Runtime::Buffer<uint8_t>(IMAGE_WIDTH, IMAGE_HEIGHT, 3);
    inRight = Halide::Runtime::Buffer<uint8_t>(IMAGE_WIDTH, IMAGE_HEIGHT, 3);
    segmented = Halide::Runtime::Buffer<uint8_t>(IMAGE_WIDTH, IMAGE_HEIGHT);
    for (int i = 0; i < IMAGE_WIDTH; i++) {
      for (int j = 0; j < IMAGE_HEIGHT; j++) {
        segmented(i, j) = rand() % 255;
        for (int k = 0; k < 3; k++) {
          inLeft(i, j, k) = rand() % 255;
          inRight(i, j, k) = rand() % 255;
        }
      }
    }
  } else {
    inLeft = Halide::Tools::load_image(args::get(left));
    inRight = Halide::Tools::load_image(args::get(right));
    segmented = Halide::Tools::load_image(args::get(seg));
  }

  uint8_t cutoff = args::get(depthCutoff);
  Halide::Runtime::Buffer<uint8_t> segmentedCutoff = Halide::Runtime::Buffer<uint8_t>::make_scalar(&cutoff);
  Halide::Runtime::Buffer<uint8_t> depthMap(inLeft.width(), inLeft.height(), inLeft.channels());
  Halide::Runtime::Buffer<uint8_t> portrait(inLeft.width(), inLeft.height(), inLeft.channels());

  if (args::get(bench)) {
    // Run a benchmark!
    Halide::Tools::BenchmarkConfig conf;
    // For faster iteration time, the benchmark shouldn't last longer than 10 seconds.
    conf.max_time = 10;
    double s = 0.f;
    if (args::get(useAuto)) {
      s = Halide::Tools::benchmark([&]() {
        portrait_gen_auto(
            inLeft,
            inRight,
            segmented,
            segmentedCutoff,
#ifdef OUTPUT_DEPTH_MAP
            depthMap,
#endif
            portrait);
      }, conf);
    } else {
      s = Halide::Tools::benchmark([&]() {
        portrait_gen(
            inLeft,
            inRight,
            segmented,
            segmentedCutoff,
#ifdef OUTPUT_DEPTH_MAP
            depthMap,
#endif
            portrait);
      }, conf);
    }
    std::cout << "Completed in " << s << " s." << std::endl;
  } else {
    // Otherwise, just run the pipeline.
    assert(portrait_gen(
        inLeft,
        inRight,
        segmented,
        segmentedCutoff,
#ifdef OUTPUT_DEPTH_MAP
        depthMap,
#endif
        portrait) == 0);
  }

  // If we weren't running with random data, then save data to output files.
  if (!args::get(randomData)) {
    // Optionally save the depth map if we were asked to.
    if (depth) {
      Halide::Tools::save_image(depthMap, args::get(depth));
    }
    Halide::Tools::save_image(portrait, args::get(output));
  }

  return 0;
}
