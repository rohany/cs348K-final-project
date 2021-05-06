// This include is generated code by Halide, so the editor might
// not be happy to include it.
#include "portrait_gen.h"

#include <iostream>

#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "halide_image_io.h"

int main() {
  // TODO (rohany): Do some command line argument parsing.
  Halide::Runtime::Buffer<uint8_t> inLeft = Halide::Tools::load_image("dude-left.jpg");
  Halide::Runtime::Buffer<uint8_t> inRight = Halide::Tools::load_image("dude-right.jpg");
  Halide::Runtime::Buffer<uint8_t> out(inLeft.width(), inLeft.height(), inLeft.channels());


  auto result = portrait_gen(inLeft, inRight, out);
  assert(result == 0);

  Halide::Tools::save_image(out, "dude-depth.jpg");

  return 0;
}
