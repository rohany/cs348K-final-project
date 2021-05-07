#include "Halide.h"

namespace {

Halide::Expr square(Halide::Expr e) {
  return e * e;
}

class Portrait : public Halide::Generator<Portrait> {
public:
    Input<Buffer<uint8_t>> inputLeft{"inputLeft", 3};
    Input<Buffer<uint8_t>> inputRight{"inputRight", 3};
    Output<Buffer<uint8_t>> depth_map{"depth_map", 3};

    // Means -size ... 0 ... size.
    int matchTileSize = 3;
    // We'll search for tiles within size pixels each way of the current pixel.
    int searchWindowSize = 20;

    // TODO (rohany): It's unlikely that I will be able to get this from the dataset,
    //  and even so, I don't think it affects the final scaled outputs anyway.
    float focalLength = 5.f;
    float stereoDistance = 5.f;

    void generate() {
      // ALGORITHM.
      Var x("x"), y("y"), c("c");

      Func cInputLeft("cInputLeft"), cInputRight("cInputRight");
      // TODO (rohany): Figure out what we can make these int sizes.
      cInputLeft(x, y, c) = cast<float_t>(Halide::BoundaryConditions::repeat_edge(inputLeft)(x, y, c));
      cInputRight(x, y, c) = cast<float_t>(Halide::BoundaryConditions::repeat_edge(inputRight)(x, y, c));

      RDom tileDiffDom(
          -matchTileSize, (2 * matchTileSize) + 1, // Over the x dimension of the tile.
          -matchTileSize, (2 * matchTileSize) + 1, // Over the y dimension of the tile.
          0, 3, // Over each color channel.
          -searchWindowSize, (2 * searchWindowSize) + 1 // Over the horizontal search window.
      );

      // Doing a sum-of-squared-differences approach. Another approach is normalized correlation.
      Func tileDiff("tileDiff");
      tileDiff(x, y, tileDiffDom.w) += square(
          cInputLeft(x + tileDiffDom.x, y + tileDiffDom.y, tileDiffDom.z) -
          // We're min-reducing the tiles along the tileDiffDom.w dimension -- i.e. taking
          // all tiles within some horizontal pixel distance from the input tile.
          cInputRight(x + tileDiffDom.x + tileDiffDom.w, y + tileDiffDom.y, tileDiffDom.z)
      );

      // TODO (rohany): Comment this.
      RDom searchWindowDom(-searchWindowSize, (2 * searchWindowSize) + 1);
      Func minTile("minTile");
      minTile(x, y) = Tuple(0, std::numeric_limits<float>::max());
      Expr minx = minTile(x, y)[0];
      Expr minSS = minTile(x, y)[1];
      Expr newx = Halide::select(minSS < tileDiff(x, y, searchWindowDom.x), minx, x + searchWindowDom.x);
      Expr newMin = Halide::min(minSS, tileDiff(x, y, searchWindowDom.x));
      minTile(x, y) = Tuple(newx, newMin);

      auto infty = std::numeric_limits<float_t>::infinity();

      Func depth("depth");
      // When the disparity is 0, map the depth to infinity.
      Expr disparity = Halide::abs(x - minTile(x, y)[0]);
      depth(x, y) = Halide::select(disparity == 0,
                                   infty,
                                   // TODO (rohany): 1.f should be focalLength * stereoDistance.
                                   1.f / cast<float_t>(disparity));

      // Scale values in depth to between 0 and 255.
      RDom imageDom(0, inputLeft.width(), 0, inputLeft.height());
      Func maxDepth("maxDepth");
      maxDepth() = 0.f;
      // Map infinity to the maximum depth.
      Expr dval = depth(imageDom.x, imageDom.y);
      maxDepth() = Halide::select(dval == infty,
                                  maxDepth(),
                                  Halide::max(maxDepth(), dval));

      Func normalizedDepth("normalizedDepth");
      auto cdepth = Halide::select(depth(x, y) == infty, maxDepth(), depth(x, y));
      // This setup makes close pixels darker.
      normalizedDepth(x, y) = 255.f - (cdepth * 255.f / maxDepth());
      // This setup makes close pixels lighter.
      // normalizedDepth(x, y) = (cdepth * 255.f / maxDepth());

      depth_map(x, y, c) = cast<uint8_t>(normalizedDepth(x, y));

      // TODO (rohany): This bilateral filter of the depth map doesn't really help. It thinks all
      //  of the depth changes are edges and doesn't blur across them. We really need
      //  instead segmentation that tells me where to blur across.
      // Apply a bilateral blur to the image.
      // float sigma = 1.5f;
      // Func guassian("guassian");
      // guassian(x, y) = exp(-((x * x + y * y) / (2.f * sigma * sigma))) / Expr(2.f * M_PI * sigma * sigma);
      // // TODO (rohany): Make these constants.
      // auto blurWindowSize = 2;
      // RDom blurDim(-blurWindowSize, 2 * blurWindowSize + 1, -blurWindowSize, 2 * blurWindowSize + 1);
      // Func depthBlurred("depthBlurred");
      // Func blurNorm("blurNorm");
      // auto weighting = [](Expr x) {
      //   // Equivalent to a gaussian blur.
      //   return 1.f;
      //   // Weights pixels that are far off in value as low.
      //   // return Halide::select(x > 30, 0.f, 1.f);
      //   // Does the same thing with a rounder effect.
      //   // float sigma = 10.f;
      //   // return exp(-x*x/(2.f*sigma*sigma)) / (sqrtf(2.f*M_PI)*sigma);
      // };
      // auto diff = Halide::abs(normalizedDepth(x, y) - normalizedDepth(x + blurDim.x, y + blurDim.y));
      // blurNorm(x, y) += weighting(diff) * guassian(blurDim.x, blurDim.y);
      // depthBlurred(x, y) += weighting(diff) *
      //                       guassian(blurDim.x, blurDim.y) * normalizedDepth(x + blurDim.x, y + blurDim.y) /
      //                       blurNorm(x, y);
      // depth_map(x, y, c) = cast<uint8_t>(depthBlurred(x, y));

      // SCHEDULE.

      // Simple schedule which is way better than the default.
      tileDiff.compute_root();
      minTile.compute_root();
      depth.compute_root();
      maxDepth.compute_root();
      normalizedDepth.compute_root();
//      blurNorm.compute_root();
//      depthBlurred.compute_root();
      depth_map.compute_root();

      depth_map.print_loop_nest();
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Portrait, portrait_gen)
