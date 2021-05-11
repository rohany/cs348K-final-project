// This include is generated code by Halide, so the editor might
// not be happy to include it.
#include "portrait_gen.h"

#include <iostream>

#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "halide_image_io.h"

#include "args.hxx"

int main(int argc, char** argv) {

  args::ArgumentParser parser("Runs the portrait mode generator.", "");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

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

  Halide::Runtime::Buffer<uint8_t> inLeft = Halide::Tools::load_image(args::get(left));
  Halide::Runtime::Buffer<uint8_t> inRight = Halide::Tools::load_image(args::get(right));
  Halide::Runtime::Buffer<uint8_t> segmented = Halide::Tools::load_image(args::get(seg));
  Halide::Runtime::Buffer<uint8_t> depthMap(inLeft.width(), inLeft.height(), inLeft.channels());
  Halide::Runtime::Buffer<uint8_t> portrait(inLeft.width(), inLeft.height(), inLeft.channels());


  auto result = portrait_gen(inLeft, inRight, segmented, depthMap, portrait);
  assert(result == 0);

  // Optionally save the depth map if we were asked to.
  if (depth) {
    Halide::Tools::save_image(depthMap, args::get(depth));
  }

  Halide::Tools::save_image(portrait, args::get(output));

  return 0;
}
