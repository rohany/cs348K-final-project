#include "Halide.h"

namespace {

class Portrait : public Halide::Generator<Portrait> {
public:
    Input<Buffer<uint8_t>> input{"input", 3};
    Output<Buffer<uint8_t>> output{"blur_y", 3};

    void generate() {
      // Dummy pipeline with a simple brighten operation.
      Var x("x"), y("y"), c("c");
      output(x, y, c) = cast<uint8_t>(Halide::min((input(x, y, c) * 3) / 2, 255));
//      output(x, y, c) = (bounds(x - 1, y - 1, c) + bounds(x, y - 1, c) + bounds(x + 1, y - 1, c) +
//                         bounds(x - 1, y, c) + (bounds(x, y, c) / 2) + bounds(x + 1, y, c) +
//                         bounds(x - 1, y + 1, c) + bounds(x, y + 1, c) + bounds(x + 1, y + 1, c)) / 9;

      output.print_loop_nest();
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Portrait, portrait_gen)
