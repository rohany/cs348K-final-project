// This include is generated code by Halide, so the editor might
// not be happy to include it.
#include "portrait_gen.h"
// For now, it declares this function:
int portrait_gen(struct halide_buffer_t *, struct halide_buffer_t *);

#include <iostream>

//#include "Halide.h"

#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "halide_image_io.h"

int main() {
  Halide::Runtime::Buffer<uint8_t> in = Halide::Tools::load_image("rohany_pic.jpg");
  Halide::Runtime::Buffer<uint8_t> out(in.width() - 8, in.height() - 2, in.channels());

  auto result = portrait_gen(in, out);
  assert(result == 0);

  Halide::Tools::save_image(out, "test_blurred.jpg");

  return 0;
}
